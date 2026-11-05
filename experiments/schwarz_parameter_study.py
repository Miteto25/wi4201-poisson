
"""WI4201 Q8: subdomain/overlap study and CG/PCG comparisons."""

import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system
from wi4201.preconditioners import (
    AdditiveSchwarzPreconditioner,
    SSORPreconditioner,
)
from wi4201.solvers import cg, pcg
from wi4201.diagnostics import measure_solver


# Fixed mesh and parameter choices.
N = 64
P_VALUES = (2, 4, 8, 16)
OVERLAPS = (0, 1, 2, 4)
TOL = 1e-10
OMEGA = 1.5

# Parameters for the fixed-mesh method comparison.
COMPARE_P = 4
COMPARE_OVERLAP = 2


def measure_cpu(operation, min_cpu=0.15, max_repeats=4096):
    """Measure average CPU time per operation using repeated trials.

    Batching avoids zero-duration timings from Windows' coarse
    process_time() resolution.
    """
    repetitions = 1

    while True:
        start = process_time()

        for _ in range(repetitions):
            value = operation()

        elapsed = process_time() - start

        if elapsed >= min_cpu or repetitions >= max_repeats:
            return value, elapsed / repetitions, repetitions

        repetitions = min(2 * repetitions, max_repeats)


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_by_overlap(rows, field, ylabel, name, folder):
    """Plot a measured quantity against overlap for each p."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for p in P_VALUES:
        group = [r for r in rows if r["p"] == p]

        ax.plot(
            [r["overlap"] for r in group],
            [r[field] for r in group],
            "o-",
            label=f"p={p}",
        )

    ax.set_xticks(OVERLAPS)
    ax.set_xlabel(
        r"Overlap layers $\ell$ (one-sided $\delta=\ell h$)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"Additive Schwarz-PCG (N={N})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(folder / name, dpi=200)
    plt.close(fig)


def main():
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)

    # Assemble once, outside all timing measurements.
    A, rhs = assemble_system(N)
    n = (N + 1)**2
    rhs_norm = np.linalg.norm(rhs)

    # Reference solution for verifying numerical correctness.
    direct = spsolve(A, rhs)

    rows = []
    comparison_M = None

    print(
        f"Q8: fixed N={N}, n={n}, "
        f"h={1/N:.6f}; tolerance={TOL:g}"
    )

    print(
        f"{'p':>3} {'ell':>4} {'delta':>9} {'K':>5} "
        f"{'setup(s)':>11} {'PCG(s)':>11} "
        f"{'sum(s)':>11} {'max local':>10}"
    )

    # Q8(a)-(c): vary subdomains and overlap.
    for p in P_VALUES:
        if N % p:
            raise ValueError(f"p={p} must divide N={N}")

        for overlap in OVERLAPS:

            # Setup includes index construction, local matrices,
            # and local sparse factorizations.
            M, setup_cpu, setup_trials = measure_cpu(
                lambda: AdditiveSchwarzPreconditioner(
                    A, N=N, p=p, overlap=overlap
                )
            )

            # PCG timing excludes setup.
            # All iterations reuse the existing preconditioner M.
            result, solve_cpu, solve_trials = measure_solver(
                lambda: pcg(
                    A, rhs,
                    preconditioner=M,
                    tol=TOL,
                    maxiter=10000,
                )
            )

            true_residual = (
                np.linalg.norm(rhs - A @ result.x) / rhs_norm
            )

            error_vs_direct = np.max(
                np.abs(result.x - direct)
            )

            if not result.converged or true_residual > TOL:
                raise RuntimeError(
                    f"AS-PCG did not converge: p={p}, ell={overlap}"
                )

            if error_vs_direct > 1e-7:
                raise RuntimeError(
                    f"AS-PCG differs from direct: "
                    f"p={p}, ell={overlap}"
                )

            row = dict(
                N=N,
                n=n,
                p=p,
                overlap=overlap,
                delta=overlap / N,
                iterations=result.iterations,
                setup_cpu_seconds=setup_cpu,
                pcg_cpu_seconds=solve_cpu,
                combined_cpu_seconds=setup_cpu + solve_cpu,
                setup_trials=setup_trials,
                pcg_trials=solve_trials,
                largest_local_size=max(
                    map(len, M.subdomain_indices)
                ),
                sum_local_sizes=sum(
                    map(len, M.subdomain_indices)
                ),
                final_relative_residual=true_residual,
                difference_from_direct_inf=error_vs_direct,
            )

            rows.append(row)

            print(
                f"{p:3d} {overlap:4d} "
                f"{row['delta']:9.6f} "
                f"{result.iterations:5d} "
                f"{setup_cpu:11.6f} "
                f"{solve_cpu:11.6f} "
                f"{setup_cpu + solve_cpu:11.6f} "
                f"{row['largest_local_size']:10d}"
            )

            if (p, overlap) == (
                COMPARE_P, COMPARE_OVERLAP
            ):
                comparison_M = M

    # Save the complete parameter study.
    save_csv(
        out / "schwarz_q8_parameters.csv",
        rows,
        list(rows[0].keys()),
    )

    # Q8(a): convergence versus overlap.
    plot_by_overlap(
        rows, "iterations", "PCG iterations",
        "schwarz_q8_iterations.png", out
    )

    # Q8(c): compare setup and iteration costs.
    plot_by_overlap(
        rows, "setup_cpu_seconds", "Setup CPU time (s)",
        "schwarz_q8_setup_time.png", out
    )

    plot_by_overlap(
        rows, "pcg_cpu_seconds",
        "PCG iteration CPU time (s)",
        "schwarz_q8_pcg_time.png", out
    )

    plot_by_overlap(
        rows, "combined_cpu_seconds",
        "Setup + PCG CPU time (s)",
        "schwarz_q8_total_time.png", out
    )

    # Q8(d): compare CG, SSOR-PCG, and AS-PCG.
    ssor_M = SSORPreconditioner(A, omega=OMEGA)

    methods = (
        ("CG", lambda: cg(A, rhs, tol=TOL)),
        (
            "SSOR-PCG",
            lambda: pcg(
                A, rhs, preconditioner=ssor_M, tol=TOL
            ),
        ),
        (
            "AS-PCG",
            lambda: pcg(
                A, rhs, preconditioner=comparison_M, tol=TOL
            ),
        ),
    )

    comparison = []
    fig, ax = plt.subplots(figsize=(7, 4.5))

    print(
        f"\nQ8(d): comparison at N={N}, "
        f"AS p={COMPARE_P}, ell={COMPARE_OVERLAP}"
    )

    print(
        f"{'method':>10} {'K':>6} "
        f"{'PCG CPU(s)':>12} {'true residual':>15}"
    )

    for name, run in methods:
        result, cpu, repeats = measure_solver(run)

        relative = (
            np.linalg.norm(rhs - A @ result.x) / rhs_norm
        )

        if not result.converged or relative > TOL:
            raise RuntimeError(
                f"{name} did not satisfy the tolerance"
            )

        comparison.append(dict(
            method=name,
            N=N,
            n=n,
            iterations=result.iterations,
            iteration_cpu_seconds=cpu,
            timing_trials=repeats,
            true_relative_residual=relative,
        ))

        ax.semilogy(
            np.arange(result.iterations + 1),
            result.relative_residuals,
            label=name,
        )

        print(
            f"{name:>10} {result.iterations:6d} "
            f"{cpu:12.6f} {relative:15.3e}"
        )

    # Save method comparison plot.
    ax.axhline(
        TOL, color="black",
        linestyle="--", label="Tolerance"
    )

    ax.set_xlabel("CG/PCG iteration")
    ax.set_ylabel(
        r"True relative residual "
        r"$\|f_h-A_hu_{h,k}\|_2/\|f_h\|_2$"
    )
    ax.set_title(
        fr"Fixed mesh $N={N}$; "
        fr"AS: $p={COMPARE_P}$, "
        fr"$\ell={COMPARE_OVERLAP}$"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        out / "schwarz_q8_method_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    save_csv(
        out / "schwarz_q8_comparison.csv",
        comparison,
        list(comparison[0].keys()),
    )

    print(f"\nSaved Q8 tables and figures in: {out}")
    print(
        "CPU timer: time.process_time(), "
        "averaged over repeated operations"
    )
    print(
        "Setup and PCG iteration CPU times "
        "are measured separately."
    )


if __name__ == "__main__":
    main()

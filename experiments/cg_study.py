
"""WI4201 implementation Question 6: CG and SSOR-preconditioned CG."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system
from wi4201.preconditioners import SSORPreconditioner
from wi4201.solvers import cg, pcg, ssor
from wi4201.diagnostics import averaged_cpu_run


def main():
    # Use the same meshes as in the previous SSOR experiments.
    N_values = [8, 16, 32, 64]
    omega = 1.5
    tol = 1e-10
    maxiter = 10000

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    print(f"{'N':>4} {'n':>7} {'method':>10} {'K':>6} "
          f"{'CPU (s)':>12} {'true residual':>16} {'diff vs direct':>16}")

    for N, ax in zip(N_values, axes.flat):
        # Neither matrix assembly nor SSOR preconditioner setup is timed.
        A, rhs = assemble_system(N)
        M = SSORPreconditioner(A, omega=omega)
        direct = spsolve(A, rhs)  # Validation; not part of CPU timing.
        n = (N + 1)**2

        
        methods = {
            "SSOR": averaged_cpu_run(
                lambda: ssor(
                    A, rhs, omega=omega, tol=tol, maxiter=maxiter
                )
            ),
            "CG": averaged_cpu_run(
                lambda: cg(
                    A, rhs, tol=tol, maxiter=maxiter
                )
            ),
            "SSOR-PCG": averaged_cpu_run(
                lambda: pcg(
                    A, rhs, preconditioner=M,
                    tol=tol, maxiter=maxiter
                )
            ),
        }


        with (out / f"cg_residuals_N{N}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["method", "iteration", "relative_residual"])
            for name, result in methods.items():
                if not result.converged:
                    raise RuntimeError(f"{name} failed to converge for N={N}")

                # Validate that the reported stopping residual is genuine.
                true_relative = np.linalg.norm(rhs - A @ result.x) / np.linalg.norm(rhs)
                if true_relative > tol:
                    raise AssertionError(f"Incorrect stopping test for {name}, N={N}")

                diff = np.max(np.abs(result.x - direct))
                print(f"{N:4d} {n:7d} {name:>10} {result.iterations:6d} "
                      f"{result.cpu_time:12.5f} {true_relative:16.3e} "
                      f"{diff:16.3e}")

                summary.append({
                    "N": N,
                    "n": n,
                    "method": name,
                    "omega": omega if name != "CG" else "",
                    "iterations": result.iterations,
                    "cpu_time_seconds": result.cpu_time,
                    "final_relative_residual": true_relative,
                    "difference_from_direct_inf": diff,
                })

                writer.writerows(
                    (name, k, float(rel))
                    for k, rel in enumerate(result.relative_residuals)
                )
                ax.semilogy(
                    np.arange(result.iterations + 1),
                    result.relative_residuals,
                    label=name,
                )

        ax.axhline(tol, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(fr"$N={N}$, $n={n}$")
        ax.set_xlabel("Iteration k")
        ax.set_ylabel("True relative residual")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend()

    fig.suptitle(r"SSOR vs CG vs SSOR-PCG ($\omega=1.5$)")
    fig.tight_layout()
    fig.savefig(out / "cg_comparison.png", dpi=200)
    plt.close(fig)

    # Save an easily reusable table for the report.
    columns = [
        "N", "n", "method", "omega", "iterations",
        "cpu_time_seconds", "final_relative_residual",
        "difference_from_direct_inf",
    ]
    with (out / "cg_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    # Compare CPU times between methods for all problem sizes.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("SSOR", "CG", "SSOR-PCG"):
        rows = [row for row in summary if row["method"] == name]
        ax.plot(
            [row["n"] for row in rows],
            [row["cpu_time_seconds"] for row in rows],
            marker="o", label=name,
        )
    ax.set_xlabel("Number of unknowns n")
    ax.set_ylabel("CPU time to convergence (s)")
    ax.set_title("CPU time: SSOR, CG and SSOR-PCG")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cg_cpu_comparison.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved to {out}:")
    print("  cg_comparison.png, cg_cpu_comparison.png")
    print("  cg_summary.csv and cg_residuals_N*.csv")
    print("Timer: time.process_time(); excludes assembly and preconditioner setup")


if __name__ == "__main__":
    main()

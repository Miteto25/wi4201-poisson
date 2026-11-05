
"""WI4201 implementation Q7: verify additive Schwarz-preconditioned CG."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system
from wi4201.preconditioners import AdditiveSchwarzPreconditioner
from wi4201.solvers import pcg
from wi4201.diagnostics import averaged_cpu_run, measure_cpu


def main():
    N = 32
    p = 4
    overlap = 2
    tol = 1e-10

    # Assembly is excluded from timing.
    A, rhs = assemble_system(N)

    # Construct subdomains and factorize local matrices once.
    M, setup_cpu, _ = measure_cpu(
        lambda: AdditiveSchwarzPreconditioner(A, N=N, p=p, overlap=overlap)
    )

    # Existing PCG solver, with the additive Schwarz preconditioner.
    result = averaged_cpu_run(lambda: pcg(
        A, rhs, preconditioner=M,
        tol=tol, maxiter=10000
    ))

    # Compare with the sparse direct solution.
    reference = spsolve(A, rhs)

    true_residual = (
        np.linalg.norm(rhs - A @ result.x)
        / np.linalg.norm(rhs)
    )
    difference = np.max(np.abs(result.x - reference))

    print(
        f"N={N}, n={(N+1)**2}, p={p}, "
        f"overlap={overlap}, delta={M.delta:.6f}"
    )
    print("Owned column ranges:", M.owned_column_ranges)
    print("Extended column ranges:", M.extended_column_ranges)
    print(
        "Local problem sizes:",
        [len(ids) for ids in M.subdomain_indices]
    )
    print(f"Setup CPU time:              {setup_cpu:.6f} s")
    print(f"PCG iteration CPU time:      {result.cpu_time:.6f} s")
    print(f"PCG iterations:              {result.iterations}")
    print(f"Final true relative residual: {true_residual:.6e}")
    print(f"Max difference vs direct:    {difference:.6e}")

    # Numerical verification required by Question 7.
    if not result.converged or true_residual > tol:
        raise AssertionError(
            "Additive Schwarz-PCG did not satisfy the tolerance"
        )

    if difference >= 1e-8:
        raise AssertionError(
            "AS-PCG and the sparse direct solution disagree"
        )

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a summary of the experiment.
    columns = [
        "N", "n", "p", "overlap", "delta", "iterations",
        "setup_cpu_seconds", "pcg_cpu_seconds",
        "final_true_relative_residual", "difference_from_direct_inf",
    ]

    with (output_dir / "schwarz_q7_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "N": N,
            "n": (N + 1)**2,
            "p": p,
            "overlap": overlap,
            "delta": M.delta,
            "iterations": result.iterations,
            "setup_cpu_seconds": setup_cpu,
            "pcg_cpu_seconds": result.cpu_time,
            "final_true_relative_residual": true_residual,
            "difference_from_direct_inf": difference,
        })

    # Save the true relative residual after every iteration.
    with (output_dir / "schwarz_q7_residuals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "true_relative_residual"])
        writer.writerows(enumerate(result.relative_residuals))

    # Plot convergence.
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.semilogy(
        range(result.iterations + 1),
        result.relative_residuals,
        marker=".",
        label="AS-PCG"
    )
    ax.axhline(
        tol,
        linestyle="--",
        color="black",
        label=r"Tolerance $10^{-10}$"
    )

    ax.set_xlabel("PCG iteration")
    ax.set_ylabel(
        r"$\|f_h-A_hu_{h,k}\|_2/\|f_h\|_2$"
    )
    ax.set_title(
        fr"Additive Schwarz-PCG ($N={N}$, "
        fr"$p={p}$, $\ell={overlap}$)"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "schwarz_q7_convergence.png",
        dpi=200
    )
    plt.close(fig)

    print(f"Saved Q7 CSV files and plot to: {output_dir}")


if __name__ == "__main__":
    main()

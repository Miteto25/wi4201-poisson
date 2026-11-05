
"""WI4201 implementation Questions 3--5: SSOR convergence and CPU cost."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wi4201.discretization import assemble_system
from wi4201.solvers import ssor
from wi4201.diagnostics import averaged_cpu_run


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)


def main():
    N_values = [8, 16, 32, 64]
    omega = 1.5
    tol = 1e-10

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    factors = []
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Q3 and Q5: run SSOR, record iterations and CPU time.
    print(f"{'N':>5} {'n':>8} {'K':>7} {'CPU (s)':>12} {'final residual':>17}")

    for N in N_values:
        A, rhs = assemble_system(N)
        result = averaged_cpu_run(
            lambda: ssor(A, rhs, omega=omega, tol=tol, maxiter=10000)
        )

        if not result.converged:
            raise RuntimeError(f"SSOR failed to converge for N={N}")

        n = (N + 1) ** 2
        K = result.iterations
        residuals = result.relative_residuals

        print(f"{N:5d} {n:8d} {K:7d} {result.cpu_time:12.4f} "
              f"{residuals[-1]:17.4e}")

        summary.append(dict(
            N=N, n=n, omega=omega, iterations=K,
            cpu_time_seconds=result.cpu_time,
            final_relative_residual=float(residuals[-1]),
            n_times_K=n * K,
            cpu_seconds_per_nK=result.cpu_time / (n * K),
        ))

        # Q3: save residual history.
        with (out / f"ssor_residuals_N{N}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "relative_residual"])
            writer.writerows(enumerate(residuals))

        ax.semilogy(
            range(K + 1), residuals,
            label=fr"$N={N}$, $n={n}$"
        )

        # Q4: final five residual reduction factors.
        ratios = residuals[-5:] / residuals[-6:-1]
        estimate = float(np.exp(np.mean(np.log(ratios))))

        factors.append(dict(
            N=N, n=n, iterations=K,
            ratio_K_minus_4=float(ratios[0]),
            ratio_K_minus_3=float(ratios[1]),
            ratio_K_minus_2=float(ratios[2]),
            ratio_K_minus_1=float(ratios[3]),
            ratio_K=float(ratios[4]),
            estimated_factor=estimate,
        ))

    # Q3: convergence plot.
    ax.axhline(
        tol, color="black", linestyle="--", linewidth=1,
        label=r"Tolerance $10^{-10}$"
    )
    ax.set_xlabel("Complete SSOR iteration k (forward + backward)")
    ax.set_ylabel(
        r"Relative residual $\|f_h-A_hu_{h,k}\|_2/\|f_h\|_2$"
    )
    ax.set_title(r"SSOR convergence ($\omega=1.5$)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "ssor_convergence.png", dpi=200)
    plt.close(fig)

    save_csv(out / "ssor_summary.csv", summary, [
        "N", "n", "omega", "iterations", "cpu_time_seconds",
        "final_relative_residual",
    ])

    # Q4: print and save final five reduction factors.
    print("\nQ4: final five residual reduction factors")
    print(f"{'N':>5} {'K-4':>10} {'K-3':>10} {'K-2':>10} "
          f"{'K-1':>10} {'K':>10} {'estimate':>11}")

    for row in factors:
        print(f"{row['N']:5d} " + " ".join(
            f"{row[key]:10.6f}" for key in [
                "ratio_K_minus_4", "ratio_K_minus_3",
                "ratio_K_minus_2", "ratio_K_minus_1",
                "ratio_K", "estimated_factor",
            ]
        ))

    save_csv(out / "ssor_asymptotic_factors.csv", factors, [
        "N", "n", "iterations", "ratio_K_minus_4",
        "ratio_K_minus_3", "ratio_K_minus_2",
        "ratio_K_minus_1", "ratio_K", "estimated_factor",
    ])

    # Q5: CPU timing and theoretical work comparison.
    print("\nQ5: CPU timing and sparse-work estimate")
    print(f"{'N':>5} {'n':>8} {'K':>7} {'CPU (s)':>12} "
          f"{'n*K':>12} {'CPU/(n*K)':>14}")

    for row in summary:
        print(
            f"{row['N']:5d} {row['n']:8d} "
            f"{row['iterations']:7d} "
            f"{row['cpu_time_seconds']:12.4f} "
            f"{row['n_times_K']:12d} "
            f"{row['cpu_seconds_per_nK']:14.3e}"
        )

    save_csv(out / "ssor_cost.csv", summary, [
        "N", "n", "iterations", "cpu_time_seconds",
        "n_times_K", "cpu_seconds_per_nK",
    ])

    # Q5: log-log comparison of CPU time with O(n*K).
    work = np.array(
        [row["n_times_K"] for row in summary], dtype=float
    )
    cpu = np.array(
        [row["cpu_time_seconds"] for row in summary]
    )

    if np.all(cpu > 0):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        ax.loglog(
            work, cpu, "o-", label="Measured CPU time"
        )

        # Reference proportional to n*K, anchored at largest grid.
        ax.loglog(
            work, cpu[-1] * work / work[-1], "--",
            label=r"Reference $T\propto nK$"
        )

        ax.set_xlabel(r"Work proxy $nK$")
        ax.set_ylabel("CPU time (s)")
        ax.set_title("SSOR CPU cost vs. sparse-work estimate")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend()

        fig.tight_layout()
        fig.savefig(out / "ssor_cpu_cost.png", dpi=200)
        plt.close(fig)

    print(f"\nResults saved to: {out}")
    print("Q4: ssor_asymptotic_factors.csv")
    print("Q5: ssor_cost.csv and ssor_cpu_cost.png")
    print("CPU timer: time.process_time() in solvers.py")


if __name__ == "__main__":
    main()

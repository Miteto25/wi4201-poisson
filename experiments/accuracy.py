
"""Implementation Question 2: verify second-order spatial accuracy."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system, grid_points
from wi4201.problem import exact_solution


def main():
    # h = 2**(-q), so N = 2**q.
    q_values = range(3, 8)  # q = 3, 4, 5, 6, 7
    results = []

    for q in q_values:
        N = 2**q
        h = 1.0 / N

        # Assemble the sparse linear system.
        A, rhs = assemble_system(N)

        # Solve using a sparse direct solver.
        u_h = spsolve(A, rhs)

        # Evaluate the exact solution at every grid node.
        x, y = grid_points(N)
        u_exact = exact_solution(x, y)

        # Infinity norm: maximum absolute nodal error.
        error = np.max(np.abs(u_h - u_exact))

        results.append({
            "q": q,
            "N": N,
            "n": (N + 1) ** 2,
            "h": h,
            "error_inf": error,
        })

    # Observed convergence order between consecutive grids.
    for i, row in enumerate(results):
        row["order"] = (
            np.nan if i == 0 else
            np.log(results[i - 1]["error_inf"] / row["error_inf"])
            / np.log(results[i - 1]["h"] / row["h"])
        )

    # Print the results table.
    print(f"{'q':>3} {'N':>5} {'n':>8} {'h':>12} "
          f"{'max error':>15} {'order':>8}")

    for row in results:
        order_text = (
            "---" if np.isnan(row["order"])
            else f"{row['order']:.4f}"
        )
        print(f"{row['q']:3d} {row['N']:5d} {row['n']:8d} "
              f"{row['h']:12.6f} {row['error_inf']:15.8e} "
              f"{order_text:>8}")

    # Fit a straight line in log-log space.
    h_values = np.array([r["h"] for r in results])
    errors = np.array([r["error_inf"] for r in results])

    fitted_order = np.polyfit(
        np.log(h_values), np.log(errors), deg=1
    )[0]

    print(f"\nObserved order (log-log fit): {fitted_order:.4f}")

    # Create the output directory.
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical results as CSV.
    csv_path = output_dir / "discretization_error.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["q", "N", "n", "h", "error_inf", "order"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Plot error against grid spacing on logarithmic axes.
    reference = errors[0] * (h_values / h_values[0]) ** 2

    plt.figure(figsize=(6.5, 4.5))
    plt.loglog(h_values, errors, "o-", label="Measured error")
    plt.loglog(h_values, reference, "--", label=r"Reference $O(h^2)$")

    plt.xlabel(r"Grid spacing $h$")
    plt.ylabel(r"$\|u_h-u_{\mathrm{ex}}^h\|_\infty$")
    plt.title("Poisson equation: discretization error")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "discretization_error.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved table: {csv_path}")
    print(f"Saved figure: {plot_path}")


if __name__ == "__main__":
    main()

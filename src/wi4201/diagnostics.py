"""Repeated CPU measurements using the solvers' iteration-only timers."""

from dataclasses import replace
from time import process_time


def measure_cpu(operation, min_cpu=0.25, max_repeats=4096):
    """Batch setup operations to resolve short CPU durations."""
    repetitions = 1
    while True:
        start = process_time()
        for _ in range(repetitions):
            value = operation()
        elapsed = process_time() - start
        if elapsed >= min_cpu or repetitions >= max_repeats:
            return value, elapsed / repetitions, repetitions
        repetitions = min(2 * repetitions, max_repeats)


def measure_solver(solve, min_cpu=0.25, max_runs=5000):
    """Return (last result, mean iteration CPU seconds, number of trials).

    Each call starts from zero. Sum only SolverResult.cpu_time, so matrix
    splitting, assembly and preconditioner construction are not charged.
    Accumulating CPU time reduces Windows timer quantisation for short runs.
    """
    total = 0.0
    runs = 0
    while total < min_cpu and runs < max_runs:
        result = solve()
        if not result.converged:
            raise RuntimeError("Cannot benchmark a solve that did not converge")
        total += result.cpu_time
        runs += 1
    return result, total / runs, runs


def averaged_cpu_run(solve):
    """Return a solver result carrying the mean iteration CPU time."""
    result, cpu, _ = measure_solver(solve)
    return replace(result, cpu_time=cpu)

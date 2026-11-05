"""Timing must exclude setup even when the callable takes longer overall."""

import numpy as np

from wi4201.diagnostics import measure_solver
from wi4201.solvers import SolverResult


def test_iteration_timer_is_averaged_instead_of_outer_duration():
    durations = iter([0.1, 0.2])

    def solve():
        return SolverResult(np.zeros(1), 1, np.array([1.0, 0.0]), next(durations), True)

    _, cpu, runs = measure_solver(solve, min_cpu=0.25)
    assert runs == 2
    assert np.isclose(cpu, 0.15)

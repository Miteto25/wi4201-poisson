
"""Check actual stencil coefficients and symmetric boundary elimination."""

import numpy as np

from wi4201.discretization import assemble_system, grid_points
from wi4201.problem import boundary, source


def test_stencil_and_boundary_contributions():
    N = 4
    m = N + 1
    A, rhs = assemble_system(N)
    x, y = grid_points(N)
    edge = (x == 0) | (x == 1) | (y == 0) | (y == 1)
    dense = A.toarray()
    # Central point: four interior neighbours, hence no elimination.
    k = 2 + 2 * m
    assert dense[k, k] == 4 * N**2
    assert np.all(dense[k, [k-1, k+1, k-m, k+m]] == -N**2)
    assert np.count_nonzero(dense[k]) == 5
    # Boundary rows and columns must BOTH be identity equations.
    assert np.array_equal(dense[edge], np.eye(m*m)[edge])
    assert np.array_equal(dense[:, edge], np.eye(m*m)[:, edge])
    assert np.allclose(rhs[edge], boundary(x[edge], y[edge]))
    # Point (3/4,3/4) touches the top and right prescribed boundaries.
    k = 3 + 3 * m
    expected = source(x[k], y[k]) + N**2 * (
        boundary(1, y[k]) + boundary(x[k], 1)
    )
    assert np.isclose(rhs[k], expected)


import numpy as np
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system, grid_points
from wi4201.problem import exact_solution
from wi4201.solvers import ssor, cg, pcg
from wi4201.preconditioners import SSORPreconditioner


def test_matrix_is_symmetric_positive_definite():
    A, _ = assemble_system(4)

    assert A.shape == (25, 25)
    assert (A - A.T).nnz == 0

    # Cholesky succeeds only for a symmetric positive definite matrix.
    np.linalg.cholesky(A.toarray())


def test_second_order_accuracy():
    errors = []

    for N in [8, 16]:
        A, rhs = assemble_system(N)
        u = spsolve(A, rhs)
        x, y = grid_points(N)

        error = np.max(np.abs(u - exact_solution(x, y)))
        errors.append(error)

    assert errors[1] < errors[0] / 3


def test_iterative_solvers():
    A, rhs = assemble_system(8)
    direct = spsolve(A, rhs)

    M = SSORPreconditioner(A, omega=1.5)

    results = [
        ssor(A, rhs, omega=1.5),
        cg(A, rhs),
        pcg(A, rhs, preconditioner=M),
    ]

    for result in results:
        assert result.converged
        assert len(result.relative_residuals) == result.iterations + 1

        residual = np.linalg.norm(rhs - A @ result.x)
        relative_residual = residual / np.linalg.norm(rhs)

        assert relative_residual <= 1e-10
        assert np.max(np.abs(result.x - direct)) < 1e-8

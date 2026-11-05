
"""Correctness tests for the additive Schwarz preconditioner (Q7)."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from wi4201.discretization import assemble_system
from wi4201.preconditioners import AdditiveSchwarzPreconditioner
from wi4201.solvers import pcg


def test_partition_ownership_and_overlap():
    N, p = 8, 2
    A, _ = assemble_system(N)

    M0 = AdditiveSchwarzPreconditioner(A, N, p, overlap=0)
    M1 = AdditiveSchwarzPreconditioner(A, N, p, overlap=1)

    assert M0.owned_column_ranges == [(0, 3), (4, 8)]
    assert M0.extended_column_ranges == [(0, 3), (4, 8)]
    assert M1.extended_column_ranges == [(0, 4), (3, 8)]
    assert M1.delta == 1 / N

    # Every unknown has exactly one owner with zero overlap.
    count = np.bincount(
        np.concatenate(M0.subdomain_indices),
        minlength=(N + 1)**2
    )
    assert np.all(count == 1)

    # With overlap, interface unknowns belong to both subdomains.
    overlap_count = np.bincount(
        np.concatenate(M1.subdomain_indices),
        minlength=(N + 1)**2
    )
    assert np.max(overlap_count) == 2
    assert np.min(overlap_count) == 1


def test_local_matrices_are_principal_submatrices():
    N = 8
    A, _ = assemble_system(N)

    M = AdditiveSchwarzPreconditioner(
        A, N=N, p=2, overlap=1
    )

    for ids, local_A in zip(
        M.subdomain_indices, M.local_matrices
    ):
        expected = A[ids, :][:, ids]
        assert (local_A - expected).nnz == 0


def test_single_subdomain_equals_direct_inverse():
    N = 8
    A, rhs = assemble_system(N)

    M = AdditiveSchwarzPreconditioner(
        A, N=N, p=1, overlap=0
    )

    assert np.allclose(
        M.solve(rhs),
        spsolve(A, rhs),
        atol=1e-12
    )


def test_as_pcg_converges_to_direct_solution():
    N = 16
    A, rhs = assemble_system(N)

    M = AdditiveSchwarzPreconditioner(
        A, N=N, p=4, overlap=2
    )

    result = pcg(
        A, rhs,
        preconditioner=M,
        tol=1e-10,
        maxiter=10000
    )

    assert result.converged

    assert result.iterations == (
        len(result.relative_residuals) - 1
    )

    true_residual = (
        np.linalg.norm(rhs - A @ result.x)
        / np.linalg.norm(rhs)
    )
    assert true_residual <= 1e-10

    direct_solution = spsolve(A, rhs)

    assert np.max(
        np.abs(result.x - direct_solution)
    ) < 1e-8


@pytest.mark.parametrize(
    "N,p,overlap",
    [
        (8, 3, 0),   # p does not divide N
        (8, 2, -1),  # Negative overlap
        (8, 0, 0),   # Invalid number of subdomains
    ]
)
def test_invalid_partition_is_rejected(N, p, overlap):
    A, _ = assemble_system(N)

    with pytest.raises(ValueError):
        AdditiveSchwarzPreconditioner(
            A, N=N, p=p, overlap=overlap
        )

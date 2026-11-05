
"""Iterative solvers for the WI4201 discrete Poisson equation."""

from dataclasses import dataclass
from time import process_time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular


@dataclass
class SolverResult:
    """Iteration results, including the residual at iteration zero."""

    x: np.ndarray
    iterations: int
    relative_residuals: np.ndarray
    cpu_time: float
    converged: bool


def ssor(A, b, omega=1.5, tol=1e-10, maxiter=10000):
    """Solve Ax = b using symmetric successive overrelaxation.

    One complete iteration is a forward and a backward SOR sweep.
    The initial guess is zero.

    Stopping criterion: ||b - Ax||_2 / ||b||_2 <= tol.
    CPU time excludes matrix splitting/setup.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("SSOR requires 0 < omega < 2")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    A = sparse.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("Incompatible matrix/vector dimensions")

    d = A.diagonal()
    if np.any(d <= 0):
        raise ValueError("A must have a positive diagonal")

    # A = D + L + U, with actual off-diagonal entries.
    L = sparse.tril(A, k=-1, format="csr")
    U = sparse.triu(A, k=1, format="csr")
    D = sparse.diags(d, format="csr")

    # Prepare the triangular systems outside the iteration timer.
    forward_matrix = (D + omega * L).tocsr()
    backward_matrix = (D + omega * U).tocsr()

    # Required zero initial guess.
    x = np.zeros(n, dtype=float)
    norm_b = np.linalg.norm(b)

    if norm_b == 0:
        return SolverResult(x, 0, np.array([0.0]), 0.0, True)

    history = [1.0]
    converged = False
    iterations = 0

    start = process_time()

    for k in range(1, maxiter + 1):

        # Forward SOR sweep.
        forward_rhs = (
            omega * b
            + (1.0 - omega) * d * x
            - omega * (U @ x)
        )

        x_half = spsolve_triangular(
            forward_matrix, forward_rhs, lower=True
        )

        # Backward SOR sweep.
        backward_rhs = (
            omega * b
            + (1.0 - omega) * d * x_half
            - omega * (L @ x_half)
        )

        x = spsolve_triangular(
            backward_matrix, backward_rhs, lower=False
        )

        # True, unpreconditioned relative residual.
        relative_residual = np.linalg.norm(b - A @ x) / norm_b
        history.append(float(relative_residual))
        iterations = k

        if relative_residual <= tol:
            converged = True
            break

    cpu_time = process_time() - start

    return SolverResult(
        x=x,
        iterations=iterations,
        relative_residuals=np.asarray(history),
        cpu_time=cpu_time,
        converged=converged,
    )


def pcg(A, b, preconditioner=None, tol=1e-10, maxiter=10000):
    """CG / PCG with a zero initial guess and true-residual stopping.

    preconditioner=None gives ordinary CG; otherwise the object must
    provide solve(r), returning M^{-1}r for an SPD preconditioner M.

    Returns the same SolverResult type as ssor(). CPU time excludes
    matrix assembly and preconditioner construction, but includes the
    iterative operations, preconditioner applications and residual checks.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    A = sparse.csr_matrix(A)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or b.shape != (n,):
        raise ValueError("A must be square and b must have matching length")

    x = np.zeros(n, dtype=float)
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return SolverResult(x, 0, np.array([0.0]), 0.0, True)

    # The recurred residual is used by CG/PCG; b-A@x is evaluated
    # separately at each iteration for the exam's true stopping test.
    r = b.copy()
    history = [1.0]
    converged = False
    iterations = 0

    start = process_time()

    if preconditioner is None:
        z = r.copy()
    else:
        z = np.asarray(preconditioner.solve(r), dtype=float)

    if z.shape != (n,):
        raise ValueError("Preconditioner returned an incompatible vector")
    rho = float(r @ z)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("The preconditioner must be positive definite")
    p = z.copy()

    for k in range(1, maxiter + 1):
        Ap = A @ p
        denom = float(p @ Ap)
        if not np.isfinite(denom) or denom <= 0.0:
            raise ArithmeticError("CG breakdown: p.T @ A @ p must be positive")

        alpha = rho / denom
        x += alpha * p
        r -= alpha * Ap

        relative_residual = np.linalg.norm(b - A @ x) / b_norm
        history.append(float(relative_residual))
        iterations = k
        if relative_residual <= tol:
            converged = True
            break

        if preconditioner is None:
            z = r.copy()
        else:
            z = np.asarray(preconditioner.solve(r), dtype=float)

        if z.shape != (n,):
            raise ValueError("Preconditioner returned an incompatible vector")
        rho_new = float(r @ z)
        if not np.isfinite(rho_new) or rho_new <= 0.0:
            raise ArithmeticError("CG breakdown: r.T @ z must be positive")

        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new

    cpu_time = process_time() - start
    return SolverResult(
        x=x,
        iterations=iterations,
        relative_residuals=np.asarray(history),
        cpu_time=cpu_time,
        converged=converged,
    )


def cg(A, b, tol=1e-10, maxiter=10000):
    """Unpreconditioned conjugate gradient (same interface as PCG)."""
    return pcg(A, b, preconditioner=None, tol=tol, maxiter=maxiter)

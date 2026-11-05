
"""Preconditioners for the WI4201 discrete Poisson equation."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular


class SSORPreconditioner:
    """Apply the inverse of the symmetric SSOR preconditioner.

    For A = D + L + U (L, U are actual strictly triangular parts),

        M = (D + omega*L) D^{-1} (D + omega*U) / [omega*(2-omega)].

    The matrix M is never formed or inverted. Its triangular factors
    are built once, and solve(r) returns M^{-1} r.
    """

    def __init__(self, A, omega=1.5):
        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR requires 0 < omega < 2")

        A = sparse.csr_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")

        self.d = A.diagonal().copy()
        if np.any(self.d <= 0):
            raise ValueError("A must have a strictly positive diagonal")

        L = sparse.tril(A, k=-1, format="csr")
        U = sparse.triu(A, k=1, format="csr")
        D = sparse.diags(self.d, format="csr")

        self.lower = (D + omega * L).tocsr()
        self.upper = (D + omega * U).tocsr()
        self.scale = omega * (2.0 - omega)

    def solve(self, r):
        """Compute z = M^{-1} r using a forward and backward solve."""
        r = np.asarray(r, dtype=float)
        if r.shape != self.d.shape:
            raise ValueError("Residual has the wrong shape")

        y = spsolve_triangular(self.lower, r, lower=True)
        z = spsolve_triangular(
            self.upper, self.d * y, lower=False
        )
        return self.scale * z


class AdditiveSchwarzPreconditioner:
    """One-level overlapping additive Schwarz for the 2D Poisson matrix.

    Divide the N+1 grid columns into p *owned* vertical strips.
    p must divide N. For width w=N//p, columns at an internal
    interface x=i*w belong to the strip on the RIGHT. Column N
    belongs to the last strip. This assigns every unknown once when
    overlap=0, including the physical-boundary unknowns.

    Each strip is extended by `overlap` columns on both available
    sides, clipped to 0,...,N. With zero-based ordering, global
    grid node (i,j) has index i+j*(N+1).

    Setup builds R_i A R_i.T implicitly via sparse indexing and
    factors each local matrix once. solve(r) computes the sum of
    prolongated local solutions without forming any matrix inverse.
    """

    def __init__(self, A, N: int, p: int, overlap: int = 0):
        from time import process_time

        start = process_time()

        for name, value in (("N", N), ("p", p), ("overlap", overlap)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be an integer")
        if N < 1:
            raise ValueError("N must be positive")
        if p < 1 or N % p != 0:
            raise ValueError("p must be positive and divide N")
        if overlap < 0:
            raise ValueError("overlap must be nonnegative")

        A = sparse.csr_matrix(A)
        n = (N + 1) ** 2
        if A.shape != (n, n):
            raise ValueError(f"A must have shape ({n}, {n}) for N={N}")

        self.N = int(N)
        self.p = int(p)
        self.overlap = int(overlap)
        self.delta = overlap / N
        self.n = n

        self.subdomain_indices = []
        self.local_matrices = []
        self.local_factorizations = []
        self.owned_column_ranges = []
        self.extended_column_ranges = []

        width = N // p
        y_rows = np.arange(N + 1, dtype=np.intp)[:, None]

        for strip in range(p):
            # Internal interface belongs to the strip on the right.
            owned_left = strip * width
            owned_right = (strip + 1) * width - 1

            if strip == p - 1:
                owned_right = N

            left = max(0, owned_left - overlap)
            right = min(N, owned_right + overlap)

            self.owned_column_ranges.append((owned_left, owned_right))
            self.extended_column_ranges.append((left, right))

            x_columns = np.arange(
                left, right + 1, dtype=np.intp
            )[None, :]

            indices = (y_rows * (N + 1) + x_columns).ravel()

            # Local operator A_i = R_i A R_i.T.
            local_A = A[indices, :][:, indices].tocsc()

            # Factorize once; never compute an explicit inverse.
            factor = sparse.linalg.splu(local_A)

            self.subdomain_indices.append(indices)
            self.local_matrices.append(local_A)
            self.local_factorizations.append(factor)

        self.setup_cpu_time = process_time() - start

    def solve(self, r):
        """Return z = sum_i R_i.T A_i^{-1} R_i r."""
        r = np.asarray(r, dtype=float)

        if r.shape != (self.n,):
            raise ValueError(f"r must have shape ({self.n},)")

        z = np.zeros_like(r)

        for indices, factor in zip(
            self.subdomain_indices, self.local_factorizations
        ):
            # Restrict, solve locally, and accumulate globally.
            z[indices] += factor.solve(r[indices])

        return z

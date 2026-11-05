
"""Sparse five-point discretization of -Delta(u) = f on (0, 1)^2."""

import numpy as np
from scipy import sparse

from .problem import boundary, source


def grid_points(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened x, y coordinates, with x varying fastest.

    The zero-based index of grid node (i, j) is i + j*(N + 1).
    Both boundary and interior nodes are included.
    """
    # N counts intervals along ONE side, not the total number of unknowns.
    # Accept Python and NumPy integers, but reject True/False (Python treats
    # booleans as integers). We need at least one interval to define h = 1/N.
    if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 1:
        # Stop immediately with an explanation if the input is invalid.
        raise ValueError("N must be a positive integer")

    # Include both endpoints: for N=2 this gives [0.0, 0.5, 1.0].
    coordinates = np.linspace(0.0, 1.0, N + 1)
    # Make two (N+1)-by-(N+1) tables describing every point on the square.
    # X repeats the x coordinates across each row; Y is constant within a row.
    # For N=2: X=[[0,.5,1],[0,.5,1],[0,.5,1]],
    #          Y=[[0,0,0],[.5,.5,.5],[1,1,1]].
    X, Y = np.meshgrid(coordinates, coordinates, indexing="xy")
    # Flatten each table row by row into a one-dimensional array.
    # Matching entries x[k], y[k] describe the SAME grid point.
    # Thus we walk along x first, then move up to the next y row.
    return X.ravel(), Y.ravel()


def assemble_system(N: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble A_h u_h = f_h with Dirichlet nodes retained.

    The 5-point Laplacian is built from a 1D tridiagonal operator
    via Kronecker products. Boundary rows and columns are replaced
    symmetrically by identity equations, and their contributions
    are transferred to the right-hand side.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape ((N+1)**2, (N+1)**2)
        Symmetric positive-definite matrix.
    rhs : numpy.ndarray, shape ((N+1)**2,)
        RHS including Dirichlet contributions.
    """
    # Get one x and y coordinate for each unknown; invalid N raises an error.
    x, y = grid_points(N)  # Also validates N.
    # m points per side means m*m unknowns, including the physical boundary.
    m = N + 1
    # The unit-length side is split into N equal intervals.
    h = 1.0 / N

    # First build the ONE-dimensional approximation of -d²u/dx².
    # At an interior point its row computes -u_left + 2*u_centre - u_right.
    # We apply the missing scale factor 1/h² when building the 2D matrix.
    # Build each diagonal separately with a single integer offset. This also
    # avoids Pylance's warning about multiple offsets in SciPy's type hints.
    # np.ones(k) creates k ones; off-diagonals have only m-1 entries.
    # Each matrix has m rows/columns, one for each point on a grid line.
    # CSR (compressed sparse row) stores nonzeros efficiently by row.
    T = (
        # Offset -1: put -1 below the main diagonal (left neighbour).
        sparse.diags(-np.ones(m - 1), offsets=-1, shape=(m, m), format="csr")
        # Offset 0: put 2 on the main diagonal (centre point).
        + sparse.diags(2.0 * np.ones(m), offsets=0, shape=(m, m), format="csr")
        # Offset +1: put -1 above the main diagonal (right neighbour).
        + sparse.diags(-np.ones(m - 1), offsets=1, shape=(m, m), format="csr")
    )
    # Identity matrix: ones on the diagonal and zeros elsewhere.
    I = sparse.eye(m, format="csr")

    # A Kronecker product builds a large matrix out of scaled matrix blocks.
    # kron(I,T) applies T within each fixed-y row: the x-direction derivative.
    # kron(T,I) connects matching x positions across rows: the y derivative.
    # Adding gives 4 at the centre and -1 at each of four neighbours.
    # Divide by h² to obtain the approximation to -Delta(u).
    # A_full has shape (m*m, m*m); boundary equations are corrected below.
    A_full = (sparse.kron(I, T, format="csr")
              + sparse.kron(T, I, format="csr")) / h**2

    # Build a mask: True means boundary point, False means interior point.
    boundary_grid = np.zeros((m, m), dtype=bool)
    # ':' selects all columns; -1 selects the last row. Mark bottom and top.
    boundary_grid[0, :] = boundary_grid[-1, :] = True
    # Select all rows and mark the first/last columns: left and right edges.
    boundary_grid[:, 0] = boundary_grid[:, -1] = True
    # Flatten in the SAME order as x and y so masks and unknowns line up.
    on_boundary = boundary_grid.ravel()

    # g stores prescribed boundary values, with zero placeholders inside.
    g = np.zeros(m * m)
    # Boolean indexing selects only boundary coordinates and entries.
    # Evaluate b(x,y)=sin(x*y) there and store the resulting known values.
    g[on_boundary] = boundary(x[on_boundary], y[on_boundary])

    # source(x,y) evaluates f at all grid points. '@' is matrix-vector
    # multiplication: A_full @ g collects contributions from known values.
    # Subtract those contributions because their matrix columns are removed
    # below. For a neighbour with value g_j and coefficient -1/h², this adds
    # g_j/h² to the interior right-hand side.
    rhs = source(x, y) - A_full @ g
    # At boundary points we want u_j = g_j, rather than a PDE equation.
    # Their final matrix rows will be identity rows, so their RHS must be g_j.
    rhs[on_boundary] = g[on_boundary]

    # '~' reverses the Boolean mask: True now means interior.
    # astype(float) turns True/False into 1.0/0.0; sparse.diags puts these
    # values on a diagonal. P keeps interior entries and removes boundary ones.
    P = sparse.diags((~on_boundary).astype(float), format="csr")
    # B does the opposite: its diagonal is 1 on the boundary and 0 inside.
    B = sparse.diags(on_boundary.astype(float), format="csr")
    # Left multiplication by P clears boundary ROWS; right multiplication
    # clears boundary COLUMNS. Clearing both preserves symmetry.
    # Adding B inserts the boundary equations 1*u_j = g_j.
    # The interior Dirichlet Laplacian is positive definite, and boundary
    # entries contribute positive squares, so the full matrix is also SPD.
    # Convert the result to CSR for efficient matrix-vector multiplication.
    A = (P @ A_full @ P + B).tocsr()
    # Remove stored entries that became zero during the matrix operations.
    # This changes storage only, not the mathematical matrix.
    A.eliminate_zeros()

    # Return the assembled system; no linear solver has been called yet.
    return A, rhs

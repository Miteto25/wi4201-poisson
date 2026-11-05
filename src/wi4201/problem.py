
"""Manufactured exact solution for the WI4201 Poisson problem."""

import numpy as np


def exact_solution(x, y):
    """Exact solution u(x, y) = sin(x*y), for scalars or NumPy arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sin(x * y)


def source(x, y):
    """Source f = -Delta(u) = (x**2 + y**2)*sin(x*y)."""
    # Convert x to a floating-point NumPy array so the same formula works
    # for a single coordinate, a list, or an entire array of grid coordinates.
    x = np.asarray(x, dtype=float)
    # Do the same for y; matching x and y entries describe a grid point.
    y = np.asarray(y, dtype=float)
    # For u=sin(x*y), u_xx=-y²*sin(x*y) and u_yy=-x²*sin(x*y).
    # Therefore f=-(u_xx+u_yy)=(x²+y²)*sin(x*y).
    # '**2' squares each entry; '*' multiplies element by element;
    # np.sin applies sine (in radians) to each entry.
    return (x**2 + y**2) * np.sin(x * y)


def boundary(x, y):
    """Dirichlet data b = u on the boundary of the unit square."""
    # Boundary values must match the exact solution sin(x*y).
    # Reuse exact_solution rather than writing the same formula again.
    # The caller selects boundary points; this function does not select them.
    # This gives 0 on x=0 or y=0, sin(y) on x=1, and sin(x) on y=1.
    return exact_solution(x, y)

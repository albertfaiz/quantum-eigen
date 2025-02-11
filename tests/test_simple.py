import numpy as np
from src.eigen import solve_eigen

def test_small_grid():
    vals, _ = solve_eigen(N=5, potential='well', n_eigs=3)
    # Check that we have three eigenvalues
    assert len(vals) == 3, "Expected 3 eigenvalues for a 5x5 grid."
    # Ensure eigenvalues are in ascending order
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted in ascending order."

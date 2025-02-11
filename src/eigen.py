import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well'):
    """
    Build a discretized 2D Hamiltonian on an N x N grid.
    Parameters
    ----------
    N : int
        Number of points in each dimension (N^2 total points).
    potential : str
        Potential type: 'well' or 'harmonic'.
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
        The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N)   # grid spacing
    inv_dx2 = float(N * N)  # equivalent to 1/dx^2
    H = np.zeros((N*N, N*N), dtype=np.float64)

    # Helper function to convert 2D indices to 1D index
    def idx(i, j):
        return i * N + j

    # Potential function
    def V(i, j):
        if potential == 'well':
            return 0.  # For an infinite square well, potential is zero inside
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + y**2)  # Quadratic potential
        else:
            return 0.

    # Build the Hamiltonian matrix using finite-difference approximations
    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            # Diagonal element: kinetic + potential
            H[row, row] = -4. * inv_dx2 + V(i, j)
            # Off-diagonals: contributions from nearest neighbors
            if i > 0:
                H[row, idx(i-1, j)] = inv_dx2
            if i < N-1:
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0:
                H[row, idx(i, j-1)] = inv_dx2
            if j < N-1:
                H[row, idx(i, j+1)] = inv_dx2
    return H

def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
    """
    H = build_2d_hamiltonian(N, potential)
    # Compute eigenvalues and eigenvectors (full spectrum)
    vals, vecs = eigh(H)
    # Sort the eigenvalues and eigenvectors in ascending order
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    parser.add_argument('--N', type=int, default=10, help='Grid size in one dimension (default: 10)')
    parser.add_argument('--potential', type=str, default='well', choices=['well', 'harmonic'],
                        help="Type of potential to use: 'well' or 'harmonic' (default: well)")
    parser.add_argument('--n_eigs', type=int, default=5, help='Number of eigenvalues to output (default: 5)')
    args = parser.parse_args()

    # Sanity checks for the inputs can be added here if desired

    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print("Lowest {} eigenvalues:".format(args.n_eigs), vals)

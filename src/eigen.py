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
        # If linear boundary conditions are selected, modify the diagonal element near the boundaries.
        if args.bc == 'linear':
            # For simplicity, if on the edge, add an extra term (e.g., a*x + b*y)
            # Here we define a and b arbitrarily:
            a, b = 1.0, 1.0
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                # Compute x, y position
                x = (i - N/2) * dx
                y = (j - N/2) * dx
                H[row, row] += (a * x + b * y)

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

parser.add_argument('--bc', type=str, default='none', choices=['none', 'linear'],
                    help="Boundary condition type: 'none' for default or 'linear' for a linear function (default: none)")

    # Sanity checks for the inputs can be added here if desired

    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print("Lowest {} eigenvalues:".format(args.n_eigs), vals)
    # Save the eigenvalues to a text file
    np.savetxt(f"eigs_N{args.N}.txt", vals)
    print(f"Eigenvalues saved to eigs_N{args.N}.txt")
    
    # --- Follow-Up Exercise 1: Save Ground-State Probability Density ---
    # Check that we have computed at least one eigenvector
    if vecs.shape[1] > 0:
        # The ground state is the first eigenvector
        ground_state = vecs[:, 0]
        # Reshape it into a 2D array of size (N x N)
        ground_state_2d = ground_state.reshape((args.N, args.N))
        # Compute the probability density |ψ|^2
        psi2 = np.abs(ground_state_2d)**2
        # Save the probability density to a file
        np.savetxt(f"psi2_N{args.N}.txt", psi2)
        print(f"Ground state probability density saved to psi2_N{args.N}.txt")


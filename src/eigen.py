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
        Potential type: 'well', 'harmonic', or 'double_well'.
        
    Returns
    -------
    H : ndarray of shape (N^2, N^2)
        The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
    """
    dx = 1. / float(N)           # grid spacing
    inv_dx2 = float(N * N)       # equivalent to 1/dx^2
    H = np.zeros((N * N, N * N), dtype=np.float64)
    
    # Helper function to convert 2D indices to a 1D index
    def idx(i, j):
        return i * N + j
    
    # Potential function
    def V(i, j):
        if potential == 'well':
            return 0.
        elif potential == 'harmonic':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            return 4. * (x**2 + y**2)
        elif potential == 'double_well':
            x = (i - N/2) * dx
            y = (j - N/2) * dx
            # Define two Gaussian wells centered at (-0.3, 0) and (0.3, 0)
            well1 = np.exp(-((x + 0.3)**2 + y**2) / 0.01)
            well2 = np.exp(-((x - 0.3)**2 + y**2) / 0.01)
            return -50 * (well1 + well2)
        else:
            return 0.
    
    # Build the Hamiltonian matrix using finite-difference approximations
    for i in range(N):
        for j in range(N):
            row = idx(i, j)
            # Diagonal element: kinetic term plus potential
            H[row, row] = -4. * inv_dx2 + V(i, j)
            # Off-diagonals: contributions from nearest neighbors
            if i > 0:
                H[row, idx(i-1, j)] = inv_dx2
            if i < N - 1:
                H[row, idx(i+1, j)] = inv_dx2
            if j > 0:
                H[row, idx(i, j-1)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j+1)] = inv_dx2
    return H

def solve_eigen(N=20, potential='well', n_eigs=None):
    """
    Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
    """
    H = build_2d_hamiltonian(N, potential)
    # Compute all eigenvalues and eigenvectors
    vals, vecs = eigh(H)
    # Sort eigenvalues and corresponding eigenvectors in ascending order
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    vecs_sorted = vecs[:, idx_sorted]
    if n_eigs is None:
        return vals_sorted, vecs_sorted
    else:
        return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Solve 2D Hamiltonian eigenvalue problem.")
    parser.add_argument('--N', type=int, default=10,
                        help='Grid size in one dimension (default: 10)')
    parser.add_argument('--potential', type=str, default='well',
                        choices=['well', 'harmonic', 'double_well'],
                        help="Type of potential to use: 'well', 'harmonic', or 'double_well' (default: well)")
    parser.add_argument('--n_eigs', type=int, default=5,
                        help='Number of eigenvalues to output (default: 5)')
    args = parser.parse_args()

    # Compute eigenvalues and eigenvectors
    vals, vecs = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.n_eigs)
    print("Lowest {} eigenvalues:".format(args.n_eigs), vals)
    
    # Save eigenvalues to file
    np.savetxt(f"eigs_N{args.N}.txt", vals)
    print(f"Eigenvalues saved to eigs_N{args.N}.txt")
    
    # Save ground-state probability density (|ψ|^2) from the first eigenvector
    if vecs.shape[1] > 0:
        ground_state = vecs[:, 0]  # First eigenvector corresponds to the ground state
        ground_state_2d = ground_state.reshape((args.N, args.N))
        psi2 = np.abs(ground_state_2d)**2
        np.savetxt(f"psi2_N{args.N}.txt", psi2)
        print(f"Ground state probability density saved to psi2_N{args.N}.txt")


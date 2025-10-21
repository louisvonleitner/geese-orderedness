import numpy as np


def calculate_laplacian_matrix(matrix):
    """Using formula L = D - A to compute laplacian matrix"""

    # calculate (TODO in or out?) degrees
    degrees = matrix.sum(axis=1)

    # create degree matrix
    degree_matrix = np.diag(degrees)

    laplacian = degree_matrix - matrix

    return laplacian, degrees


def compute_algebraic_connectivity_and_connected_components(
    matrix: np.ndarray, tol=1e-12
) -> (float, int):
    """compute laplaican matrix and its eigenvalues and its number of connected components"""

    n = matrix.shape[0]

    laplacian, degrees = calculate_laplacian_matrix(matrix)

    # Initialize D^{-1/2}
    D_inv_sqrt = np.zeros((n, n))
    for i, d in enumerate(degrees):
        if d > 0:
            D_inv_sqrt[i, i] = 1.0 / np.sqrt(d)
        else:
            D_inv_sqrt[i, i] = 0.0  # isolated node

    normalized_laplacian = np.eye(n) - D_inv_sqrt @ matrix @ D_inv_sqrt

    # No. of connected components is the rank of the kernel matrix.
    # the dimension of the kernel matrix is the dimension of the matrix - its rank
    laplacian_rank = np.linalg.matrix_rank(laplacian, tol)
    kernel_rank = n - laplacian_rank

    laplacian_eigenvalues_normed = np.linalg.eigvals(normalized_laplacian)
    eigenvalues_sorted = np.sort(laplacian_eigenvalues_normed)

    algebraic_connectivity = eigenvalues_sorted[1]

    return algebraic_connectivity, kernel_rank

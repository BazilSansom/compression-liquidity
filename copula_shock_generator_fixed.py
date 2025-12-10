"""
Gaussian Copula Shock Generator

Generates correlated shocks using Gaussian copula with proper correlation control.
The parameter rho controls the dependence structure in the normal space, which
translates to dependence in uniform space.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky


def generate_gaussian_copula_shocks(num_nodes, rho, n_samples=1, seed=None, target_correlation='spearman'):
    """
    Generate correlated shocks using Gaussian copula.

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    rho : float
        Correlation coefficient (0 to 1)
        - If target_correlation='spearman': Target Spearman's rho
        - If target_correlation='pearson': Target Pearson correlation
    n_samples : int, optional
        Number of shock vectors (default: 1)
    seed : int, optional
        Random seed
    target_correlation : str, optional
        Type of correlation: 'pearson' or 'spearman' (default: 'spearman')

    Returns
    -------
    shocks : numpy.ndarray
        Uniform shocks in [0, 1] with specified correlation structure
        Shape: (n_samples, num_nodes)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert target correlation to normal copula parameter
    if target_correlation == 'spearman':
        # For Gaussian copula: Spearman's rho = (6/pi) * arcsin(rho/2)
        # Invert: rho_normal = 2 * sin(pi * spearman_rho / 6)
        normal_correlation = 2 * np.sin(np.pi * rho / 6)
    elif target_correlation == 'pearson':
        normal_correlation = rho
    else:
        raise ValueError("target_correlation must be 'pearson' or 'spearman'")

    # Build correlation matrix
    corr_matrix = np.full((num_nodes, num_nodes), normal_correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvals(corr_matrix))
    if min_eig < 0:
        corr_matrix += np.eye(num_nodes) * (-min_eig + 1e-6)

    # Generate from multivariate normal
    mean = np.zeros(num_nodes)
    try:
        normal_samples = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples)
    except np.linalg.LinAlgError:
        # Fallback to Cholesky decomposition
        L = cholesky(corr_matrix, lower=True)
        independent_normals = np.random.randn(n_samples, num_nodes)
        normal_samples = independent_normals @ L.T + mean

    # Handle shape for single sample
    if n_samples == 1:
        normal_samples = normal_samples.reshape(1, -1)

    # Transform to uniform via probability integral transform
    uniform_samples = norm.cdf(normal_samples)

    return uniform_samples

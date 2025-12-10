"""
Risk Analysis for Derivative Market Networks

This module provides tools for generating and analyzing financial networks
that mimic derivative markets, including source/sink nodes and network generation.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_array
from full_payment_algo import full_payment_algo


def generate_compression_network(N, p, num_sources, num_sinks, seed=None):
    """
    Generate a financial network with source, core, and sink nodes.

    Creates a three-tier structure:
    - Source nodes: Only outgoing edges (market makers, dealers)
    - Core nodes: Can have both incoming and outgoing edges
    - Sink nodes: Only incoming edges (end users)

    Parameters
    ----------
    N : int
        Number of core nodes
    p : float
        Probability of edge between core nodes (Erdos-Renyi parameter)
    num_sources : int
        Number of source nodes
    num_sinks : int
        Number of sink nodes
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    W : numpy.ndarray
        Weighted adjacency matrix (structure only, weights=1)
    source_nodes : list
        Indices of source nodes
    core_nodes : list
        Indices of core nodes
    sink_nodes : list
        Indices of sink nodes
    """
    if seed is not None:
        np.random.seed(seed)

    total_nodes = num_sources + N + num_sinks

    # Node indices
    source_nodes = list(range(num_sources))
    core_nodes = list(range(num_sources, num_sources + N))
    sink_nodes = list(range(num_sources + N, total_nodes))

    # Initialize adjacency matrix
    W = np.zeros((total_nodes, total_nodes))

    # Source to core connections
    for source in source_nodes:
        # Each source connects to random subset of core nodes
        num_connections = max(1, int(N * p))
        targets = np.random.choice(core_nodes, size=num_connections, replace=False)
        for target in targets:
            W[source, target] = 1.0

    # Core to core connections (Erdos-Renyi)
    for i in core_nodes:
        for j in core_nodes:
            if i != j and np.random.random() < p:
                W[i, j] = 1.0

    # Core to sink connections
    for sink in sink_nodes:
        # Each sink receives from random subset of core nodes
        num_connections = max(1, int(N * p))
        sources_to_sink = np.random.choice(core_nodes, size=num_connections, replace=False)
        for source in sources_to_sink:
            W[source, sink] = 1.0

    return W, source_nodes, core_nodes, sink_nodes


def assign_weights_to_network(L, alpha=2.0, scale=1.0):
    """
    Assign random weights to network edges.

    Uses power-law-like distribution for realistic financial network weights.

    Parameters
    ----------
    L : numpy.ndarray
        Binary adjacency matrix (0 or 1)
    alpha : float, optional
        Power-law exponent (default 2.0)
    scale : float, optional
        Scaling factor for weights (default 1.0)

    Returns
    -------
    W : numpy.ndarray
        Weighted adjacency matrix
    """
    W = L.copy()
    edges = np.where(L > 0)

    for i, j in zip(edges[0], edges[1]):
        # Generate power-law distributed weight
        # Using Pareto distribution: w ~ (1/U)^(1/alpha) where U ~ Uniform(0,1)
        u = np.random.uniform(0.001, 1.0)
        weight = scale * (1.0 / u) ** (1.0 / alpha)
        # Clip to reasonable range
        weight = min(weight, 1000 * scale)
        W[i, j] = weight

    return W


def generate_correlated_shocks(num_nodes, rho, seed=None, normalize_to_01=True):
    """
    Generate correlated shock vector using factor model.

    Uses a one-factor model: epsilon_i = sqrt(rho) * Z + sqrt(1-rho) * eps_i
    where Z is common factor, eps_i are idiosyncratic shocks.

    Parameters
    ----------
    num_nodes : int
        Number of nodes
    rho : float
        Correlation coefficient (0 to 1)
    seed : int, optional
        Random seed for reproducibility
    normalize_to_01 : bool, optional
        If True, normalize shocks to [0, 1] range (default: True)
        If False, use absolute value of normal (unbounded above)

    Returns
    -------
    epsilon : numpy.ndarray
        Shock vector (num_nodes,) with pairwise correlation rho
        If normalize_to_01=True: values in [0, 1]
        If normalize_to_01=False: values in [0, inf) (folded normal)
    """
    if seed is not None:
        np.random.seed(seed)

    # Common factor (systematic shock)
    Z = np.random.randn()

    # Idiosyncratic shocks
    eps = np.random.randn(num_nodes)

    # Correlated shocks using factor model
    epsilon = np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps

    # Make positive (shocks reduce equity)
    epsilon = np.abs(epsilon)

    # Normalize to [0, 1] if requested
    if normalize_to_01:
        # Scale to [0, 1] preserving relative magnitudes
        epsilon_min = epsilon.min()
        epsilon_max = epsilon.max()
        if epsilon_max > epsilon_min:  # Avoid division by zero
            epsilon = (epsilon - epsilon_min) / (epsilon_max - epsilon_min)
        else:
            epsilon = np.ones(num_nodes) * 0.5  # All equal, set to mid-range

    return epsilon


def compute_liability_and_equity(W, DeltaP, alpha, gamma, epsilon):
    """
    Compute liabilities and equity for network.

    Parameters
    ----------
    W : numpy.ndarray
        Network adjacency matrix
    DeltaP : float
        Price change multiplier
    alpha : float
        Buffer multiplier
    gamma : float
        Shock magnitude
    epsilon : numpy.ndarray
        Shock vector

    Returns
    -------
    L : numpy.ndarray
        Liability matrix
    e0 : numpy.ndarray
        Initial equity/cash
    """
    L = DeltaP * W
    buffer = alpha * W.sum(axis=1)
    e0 = buffer - gamma * epsilon

    return L, e0


def compute_buffer_and_shocked_equity(W, alpha, epsilon, gamma, gamma_mode='multiplicative'):
    """
    Compute buffer and equity after shock.

    Parameters
    ----------
    W : numpy.ndarray
        Network adjacency matrix
    alpha : float
        Buffer multiplier
    epsilon : numpy.ndarray
        Shock vector
    gamma : float
        Shock magnitude
    gamma_mode : str, optional
        How to apply gamma: 'multiplicative' or 'additive'

    Returns
    -------
    buffer : numpy.ndarray
        Buffer amounts
    shock : numpy.ndarray
        Shock amounts applied
    e0 : numpy.ndarray
        Shocked equity (may be negative before truncation)
    gamma_used : float
        The gamma value used (for compatibility)
    """
    buffer = alpha * W.sum(axis=1)

    if gamma_mode == 'multiplicative':
        shock = gamma * epsilon
    else:
        shock = epsilon + gamma

    e0 = buffer - shock

    return buffer, shock, e0, gamma


def run_fpa_analysis(W, DeltaP, alpha, gamma, rho, seed):
    """
    Run full payment algorithm analysis on network.

    Parameters
    ----------
    W : numpy.ndarray
        Network adjacency matrix
    DeltaP : float
        Price change multiplier
    alpha : float
        Buffer multiplier
    gamma : float
        Shock magnitude
    rho : float
        Shock correlation
    seed : int
        Random seed

    Returns
    -------
    results : dict
        Dictionary containing shortfall and other metrics
    """
    epsilon = generate_correlated_shocks(W.shape[0], rho, seed)
    L, e0 = compute_liability_and_equity(W, DeltaP, alpha, gamma, epsilon)

    p_bar_fpa, e_steady, p_steady, resid_oblig, num_iter, p_bar1 = full_payment_algo(L, e0)

    # Calculate economic shortfall (additional liquidity needed)
    # p_bar_fpa = total unpaid obligations per node
    # additional liquidity needed = max(0, unpaid - remaining_equity)
    unpaid = np.asarray(p_bar_fpa).ravel()
    e_steady_arr = np.asarray(e_steady).ravel()
    additional_liquidity_per_node = np.maximum(0, unpaid - e_steady_arr)

    shortfall = float(additional_liquidity_per_node.sum())
    total_unpaid = float(unpaid.sum())

    return {
        'shortfall': shortfall,  # Economic shortfall (additional liquidity needed)
        'total_unpaid': total_unpaid,  # Total unpaid obligations
        'unpaid': unpaid,  # Node-level unpaid obligations
        'p_bar': additional_liquidity_per_node,  # Node-level additional liquidity needed
        'e_steady': e_steady,
        'iterations': num_iter,
        'epsilon': epsilon
    }


def run_fpa_analysis_with_buffer_shock(W, DeltaP, alpha, gamma, gamma_mode, rho, seed):
    """
    Run FPA with specific buffer/shock mode.

    Parameters
    ----------
    W : numpy.ndarray
        Network adjacency matrix
    DeltaP : float
        Price change multiplier
    alpha : float
        Buffer multiplier
    gamma : float
        Shock magnitude
    gamma_mode : str
        'multiplicative' or 'additive'
    rho : float
        Shock correlation
    seed : int
        Random seed

    Returns
    -------
    results : dict
        Analysis results
    """
    epsilon = generate_correlated_shocks(W.shape[0], rho, seed)
    buffer, shock, e0, _ = compute_buffer_and_shocked_equity(W, alpha, epsilon, gamma, gamma_mode)
    L = DeltaP * W

    p_bar_fpa, e_steady, p_steady, resid_oblig, num_iter, p_bar1 = full_payment_algo(L, e0)

    # Calculate economic shortfall (additional liquidity needed)
    # p_bar_fpa = total unpaid obligations per node
    # additional liquidity needed = max(0, unpaid - remaining_equity)
    unpaid = np.asarray(p_bar_fpa).ravel()
    e_steady_arr = np.asarray(e_steady).ravel()
    additional_liquidity_per_node = np.maximum(0, unpaid - e_steady_arr)

    shortfall = float(additional_liquidity_per_node.sum())
    total_unpaid = float(unpaid.sum())

    return {
        'shortfall': shortfall,  # Economic shortfall (additional liquidity needed)
        'total_unpaid': total_unpaid,  # Total unpaid obligations
        'unpaid': unpaid,  # Node-level unpaid obligations
        'p_bar': additional_liquidity_per_node,  # Node-level additional liquidity needed
        'e_steady': e_steady,
        'iterations': num_iter,
        'epsilon': epsilon,
        'buffer': buffer
    }

"""
Network generation and basic utilities for payment obligation networks.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np

NodeType = Literal["core", "source", "sink"]


@dataclass
class PaymentNetwork:
    """
    Static network of payment obligations.

    Attributes
    ----------
    W : np.ndarray
        Adjacency matrix, shape (n_nodes, n_nodes).
        W[i, j] is the obligation from node i to node j (i pays j).
    node_types : np.ndarray
        Array of length n_nodes with entries in {"core", "source", "sink"}.
    """
    W: np.ndarray
    node_types: np.ndarray


def generate_three_tier_network(
    n_core: int,
    n_source: int,
    n_sink: int,
    rng: np.random.Generator,
    *,
    core_core_edge_prob: float = 0.5,
    source_core_edge_prob: float = 0.5,
    core_sink_edge_prob: float = 0.5,
    notional_scale: float = 1.0,
) -> PaymentNetwork:
    """
    Generate a random three-tier OTC derivatives-style network.

    The implementation details are to be filled in. This function is intended
    to be the canonical way to generate synthetic networks for the experiments.

    Parameters
    ----------
    n_core, n_source, n_sink :
        Number of nodes in each tier.
    rng :
        Numpy random Generator for reproducibility.
    core_core_edge_prob, source_core_edge_prob, core_sink_edge_prob :
        Edge probabilities for the respective blocks.
    notional_scale :
        Scale parameter for the notional size distribution.

    Returns
    -------
    PaymentNetwork
    """
    raise NotImplementedError("generate_three_tier_network is not yet implemented.")


def compute_net_positions(W: np.ndarray) -> np.ndarray:
    """
    Compute net position of each node: outgoing minus incoming obligations.

    Parameters
    ----------
    W : np.ndarray
        Adjacency matrix of obligations.

    Returns
    -------
    np.ndarray
        Length-n array of net positions.
    """
    outgoing = W.sum(axis=1)
    incoming = W.sum(axis=0)
    return outgoing - incoming

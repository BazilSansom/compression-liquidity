"""
Portfolio compression algorithms (e.g. max-C, BFF) operating on PaymentNetwork.
"""

from dataclasses import dataclass
import numpy as np

from .networks import PaymentNetwork, compute_net_positions


@dataclass
class CompressionStats:
    """
    Summary statistics for a compression run.

    Attributes
    ----------
    gross_notional_before : float
    gross_notional_after : float
    reduction_fraction : float
        (gross_before - gross_after) / gross_before
    """
    gross_notional_before: float
    gross_notional_after: float
    reduction_fraction: float


def _gross_notional(W: np.ndarray) -> float:
    """Helper to compute total gross notional."""
    return float(W.sum())


def compress_maxC(network: PaymentNetwork) -> tuple[PaymentNetwork, CompressionStats]:
    """
    Apply a 'max-C' compression algorithm to the network.

    Parameters
    ----------
    network : PaymentNetwork

    Returns
    -------
    compressed_network : PaymentNetwork
    stats : CompressionStats
    """
    # TODO: implement using CDFD-based full conservative compression
    raise NotImplementedError("compress_maxC is not yet implemented.")


def compress_BFF(network: PaymentNetwork) -> tuple[PaymentNetwork, CompressionStats]:
    """
    Apply the BFF (Balanced Flow Forwarding) compression algorithm.

    Parameters
    ----------
    network : PaymentNetwork

    Returns
    -------
    compressed_network : PaymentNetwork
    stats : CompressionStats
    """
    # TODO: implement BFF-based compression
    raise NotImplementedError("compress_BFF is not yet implemented.")

"""
Payment cascade simulation.

The core mechanical component is the Full Payment Algorithm (FPA), which takes
a network, initial buffers, and a realised shock, and computes the resulting
payment flows and liquidity shortfalls.
"""

from dataclasses import dataclass
import numpy as np

from .networks import PaymentNetwork
from .shocks import ShockModel, draw_shock


@dataclass
class SimulationOutcome:
    """
    Outcome for a single shock realisation.

    Attributes
    ----------
    total_shortfall : float
        Sum of node shortfalls.
    node_shortfalls : np.ndarray
        Length-n array with each node's shortfall.
    # (You could also add realised payments, final cash, etc., if useful.)
    """
    total_shortfall: float
    node_shortfalls: np.ndarray


def full_payment_algorithm(
    network: PaymentNetwork,
    buffers: np.ndarray,
    shock: np.ndarray,
) -> SimulationOutcome:
    """
    Full Payment Algorithm (FPA) for a single realised shock.

    This function implements the payment dynamics given:
      - a static obligation matrix W (who owes whom how much),
      - initial cash buffers b[i] at each node,
      - a realised shock s[i] (e.g. VM calls).

    Conceptually:
      1. Start from available cash b[i] - s[i] (or whatever your model uses).
      2. Iterate the payment rule (e.g. pro-rata payments subject to liquidity).
      3. Stop when a fixed point is reached (no further payments possible).
      4. Compute shortfall as obligations that remain unpaid.

    Parameters
    ----------
    network : PaymentNetwork
        Payment obligations, W[i, j] = obligation from i to j.
    buffers : np.ndarray
        Initial cash buffers b[i].
    shock : np.ndarray
        Shock vector s[i].

    Returns
    -------
    SimulationOutcome
        total_shortfall and node_shortfalls.
    """
    W = network.W
    n = W.shape[0]

    # TODO: implement the actual FPA dynamics here.
    # Placeholder implementation just returns zero shortfall:
    node_shortfalls = np.zeros(n, dtype=float)
    total_shortfall = float(node_shortfalls.sum())

    return SimulationOutcome(
        total_shortfall=total_shortfall,
        node_shortfalls=node_shortfalls,
    )


def simulate_payments_single_shock(
    network: PaymentNetwork,
    buffers: np.ndarray,
    shock: np.ndarray,
) -> SimulationOutcome:
    """
    Thin wrapper around FPA, kept for clarity / future extensions.

    Parameters
    ----------
    network : PaymentNetwork
    buffers : np.ndarray
    shock : np.ndarray

    Returns
    -------
    SimulationOutcome
    """
    return full_payment_algorithm(network, buffers, shock)



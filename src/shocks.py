"""
Shock models for variation margin (VM) or similar liquidity shocks.
"""

from dataclasses import dataclass
import numpy as np

from .networks import PaymentNetwork


@dataclass
class ShockModel:
    """
    Parameters for the distribution of shocks.

    This is intentionally generic; concrete experiments can interpret
    sigma, m, etc. as needed (e.g. m * sigma * X for portfolio size X).
    """
    sigma: float
    m: float


def draw_shock(
    rng: np.random.Generator,
    network: PaymentNetwork,
    shock_model: ShockModel,
) -> np.ndarray:
    """
    Draw a single realised shock vector for all nodes.

    The exact distribution and mapping from ShockModel to node-level shocks
    is to be specified in the experiments.

    Parameters
    ----------
    rng : np.random.Generator
    network : PaymentNetwork
    shock_model : ShockModel

    Returns
    -------
    np.ndarray
        Shock vector s[i] applied to each node's buffer.
    """
    # TODO: implement shock generation consistent with experimental design
    raise NotImplementedError("draw_shock is not yet implemented.")

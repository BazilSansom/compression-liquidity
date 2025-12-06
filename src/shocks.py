"""
Shock models for intraday liquidity drains (ξ), not VM calls.

In this framework, each node i starts the day with an intended liquidity buffer b_i.
Before the VM settlement window, nodes experience random intraday liquidity drains ξ_i
(e.g. unexpected outgoing payments, settlement flows, or other cash uses).

The effective liquidity available for settling VM is then:

    available[i] = buffers[i] - xi[i]

These intraday drains are one of the two sources of randomness in the model
(the other being random network generation).
"""

from dataclasses import dataclass
import numpy as np

from .networks import PaymentNetwork


@dataclass
class ShockModel:
    """
    Parameters for the distribution of intraday liquidity drains ξ.

    This is intentionally generic; concrete experiments can interpret sigma, m, etc.
    as needed. For example, they might control the scale and severity of ξ relative
    to some notion of 'normal' liquidity usage.

    Attributes
    ----------
    sigma : float
        Scale parameter for the dispersion of ξ.
    m : float
        Stress multiplier (e.g. number of 'sigma' moves).
    """
    sigma: float
    m: float


def draw_shock(
    rng: np.random.Generator,
    network: PaymentNetwork,
    shock_model: ShockModel,
) -> np.ndarray:
    """
    Draw a single realised intraday liquidity drain vector ξ for all nodes.

    The realised ξ reduces the buffers available at the VM settlement window via:

        available[i] = buffers[i] - xi[i]

    The exact distribution and mapping from ShockModel to node-level drains
    is to be specified in the experimental design (e.g. iid across nodes,
    correlated across nodes, scaled by node size, etc.).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    network : PaymentNetwork
        Network object (used at least for its number of nodes; potentially for
        node-specific scaling of shocks).
    shock_model : ShockModel
        Parameters governing the distribution/scale of ξ (correlation - ρ, scale - γ).

    Returns
    -------
    np.ndarray
        Shock vector xi[i] applied to each node's buffer (non-negative drains).
    """
    # TODO: implement ξ-generation consistent with the experimental design
    raise NotImplementedError("draw_shock is not yet implemented.")

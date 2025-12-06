"""
Shock models for intraday liquidity drains ξ, not VM calls.

In the model, each node i starts with an intended liquidity buffer b_i.
Before the VM settlement window, nodes experience random intraday liquidity
drains ξ_i (e.g. unexpected outgoing payments, settlement flows, or other cash uses).

The effective liquidity available for settling VM is:

    available[i] = buffers[i] - xi[i]

These intraday drains are one of the two sources of randomness in the model
(the other being random network generation). The VM shock process itself is
parameterised separately (e.g. via sigma, m); it is *not* handled here.
"""

from dataclasses import dataclass
import numpy as np

from .networks import PaymentNetwork


@dataclass
class ShockModel:
    """
    Parameters for the distribution of intraday liquidity drains ξ,
    modelled via a Gaussian copula.

    The intended structure is:
      - correlation parameter rho ∈ [0, 1), controlling the dependence of
        shocks across nodes;
      - magnitude parameter gamma ≥ 0, governing the overall stress level
        (e.g. scaling the typical size of ξ relative to some baseline).

    The exact mapping from (rho, gamma) to the distribution of ξ (including any
    normalisation or truncation) follows the specification in the paper.
    """
    rho: float   # cross-sectional correlation of ξ via Gaussian copula
    gamma: float # overall magnitude / stress parameter


def draw_shock(
    rng: np.random.Generator,
    network: PaymentNetwork,
    shock_model: ShockModel,
) -> np.ndarray:
    """
    Draw a single realised intraday liquidity drain vector ξ for all nodes,
    using a Gaussian copula with correlation rho and magnitude gamma.

    Conceptual design (to be implemented):
      1. Construct a correlation structure with parameter shock_model.rho.
      2. Draw a vector of correlated standard normals Z ~ N(0, Σ(rho)).
      3. Map Z through Φ (normal CDF) to obtain correlated uniforms U.
      4. Transform U into non-negative drains ξ, scaled by shock_model.gamma
         and possibly by node characteristics (e.g. size, outgoing VM).

    The realised ξ reduces the buffers available at the VM settlement window:

        available[i] = buffers[i] - xi[i]

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    network : PaymentNetwork
        Used at least for its number of nodes; possibly for node-specific scaling.
    shock_model : ShockModel
        Parameters (rho, gamma) governing the Gaussian-copula distribution of ξ.

    Returns
    -------
    np.ndarray
        Shock vector xi[i] applied to each node's buffer (non-negative drains).
    """
    # TODO: implement Gaussian-copula-based ξ-generation consistent with the paper
    raise NotImplementedError("draw_shock is not yet implemented.")

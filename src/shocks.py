# src/shocks.py
import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm


# ============================================================
# 1. Shock *shape* generation (dimensionless, no £ scaling)
# ============================================================

@dataclass
class ShockShapeParams:
    """
    Parameters for generating *shape* of liquidity shocks via a one-factor
    Gaussian model.

    This layer encodes dependence structure only, not magnitude.

    Interpretation
    --------------
    rho in [0,1]:
        rho = 0  → purely idiosyncratic shocks
        rho = 1  → perfectly common factor

    one_sided:
        If True, return non-negative magnitudes (liquidity drains).
        If False, return signed Gaussian factors.

    The output is dimensionless.
    """
    rho: float
    one_sided: bool = True
    mode: Literal["gaussian_factor"] = "gaussian_factor"


def generate_gaussian_factor_shapes(
    num_nodes: int,
    params: ShockShapeParams,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate correlated Gaussian shock *shapes* using a one-factor model:

        Z_i = sqrt(rho) * Z_common + sqrt(1-rho) * eps_i.

    This function generates *dimensionless* shock shapes (no scaling by V_ref).

    Parameters
    ----------
    num_nodes : int
        Number of nodes (dimension of each sample).
    params : ShockShapeParams
        One-factor correlation parameter rho and one_sided flag.
    n_samples : int, optional
        Number of independent samples to draw (default 1).
    rng : np.random.Generator, optional
        Random number generator used for sampling. If None, a fresh generator is created.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, num_nodes). If params.one_sided is True,
        values are non-negative; otherwise values are real-valued.

    """
    rho = float(params.rho)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0,1], got {rho}")

    if rng is None:
        rng = np.random.default_rng()

    z_common = rng.normal(size=(n_samples, 1))
    z_idio = rng.normal(size=(n_samples, num_nodes))

    z = np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * z_idio

    if params.one_sided:
        return np.abs(z)
    else:
        return z


def gaussian_to_uniform01(z: np.ndarray) -> np.ndarray:
    """
    Map standard normal samples to Uniform(0,1) via the normal CDF.
    """
    return norm.cdf(z)


@dataclass
class UniformShockShapeParams:
    """
    Parameters for generating correlated Uniform(0,1) shock shapes using
    a Gaussian copula.
    """
    rho: float


def generate_uniform_factor_shapes(
    num_nodes: int,
    params: UniformShockShapeParams,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate correlated Uniform(0,1) shock *shapes* using a Gaussian copula.

    This is a pure shape generator: no £ scaling and no dependence on V_ref.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (dimension of each sample).
    params : UniformShockShapeParams
        Copula correlation parameter rho.
    n_samples : int, optional
        Number of independent samples to draw (default 1).
    rng : np.random.Generator, optional
        Random number generator used for sampling. If None, a fresh generator is created.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, num_nodes) with entries in (0,1).

    """
    gauss_params = ShockShapeParams(rho=params.rho, one_sided=False)

    Z = generate_gaussian_factor_shapes(
        num_nodes=num_nodes,
        params=gauss_params,
        n_samples=n_samples,
        rng=rng,
    )

    return gaussian_to_uniform01(Z)


# ============================================================
# 2. Deterministic scaling: shape → £ liquidity drain
# ============================================================

def xi_from_uniform_base(
    V_ref: np.ndarray,
    U: np.ndarray,
    lam: float,
    *,
    scale: Literal["row_sum", "col_sum", "total"] = "row_sum",
) -> np.ndarray:
    """
    Deterministic scaling rule (common-random-numbers across λ):
        
        xi(λ) = λ * s(V_ref) ∘ U,

    where U is held fixed as λ varies.

    Parameters
    ----------
    V_ref : np.ndarray
        Reference VM obligations matrix (used only for scaling).
    U : np.ndarray
        N x 1 vector with entries in [0,1].
    lam : float
        Shock intensity parameter.
    scale : {"row_sum","col_sum","total"}
        Defines s_i(V_ref).

    Returns
    -------
    xi : np.ndarray
        N x 1 vector of non-negative liquidity drains.
    """
    V_ref = np.asarray(V_ref, dtype=float)
    U = np.asarray(U, dtype=float).reshape(-1, 1)

    N = V_ref.shape[0]
    if U.shape != (N, 1):
        raise ValueError(f"U must have shape {(N,1)}, got {U.shape}")

    eps = 1e-12
    U = np.clip(U, eps, 1.0 - eps)

    if scale == "row_sum":
        s = V_ref.sum(axis=1).reshape(-1, 1)
    elif scale == "col_sum":
        s = V_ref.sum(axis=0).reshape(-1, 1)
    elif scale == "total":
        s = np.full((N, 1), V_ref.sum() / N, dtype=float)
    else:
        raise ValueError("scale must be one of {'row_sum','col_sum','total'}")

    xi = float(lam) * s * U

    # optional but great for debugging stability
    if not np.all(np.isfinite(xi)):
        raise ValueError("xi contains non-finite values (nan/inf).")

    return xi



# ============================================================
# 3. Convenience wrapper: one-shot realised £ shock
# ============================================================

def generate_liquidity_drain_xi(
    V_ref: np.ndarray,
    *,
    rho: float,
    lam: float,
    rng: np.random.Generator | None = None,
    scale: Literal["row_sum", "col_sum", "total"] = "row_sum",
) -> np.ndarray:
    """

      Convenience wrapper: generate a realised £-valued liquidity drain vector xi.

    Steps:
      1) draw a correlated Uniform(0,1) shape U using a Gaussian copula
      2) apply deterministic scaling via xi_from_uniform_base

    Parameters
    ----------
    V_ref : np.ndarray
        Reference VM obligations matrix used only for scaling.
    rho : float
        Copula correlation parameter in [0,1].
    lam : float
        Shock intensity.
    rng : np.random.Generator, optional
        RNG used for the shape draw. If None, a fresh generator is created.
    scale : {"row_sum","col_sum","total"}, optional
        Scaling rule s(V_ref) used in xi_from_uniform_base.

    Returns
    -------
    np.ndarray
        N x 1 vector xi of non-negative liquidity drains.

    """
    if rng is None:
        rng = np.random.default_rng()

    V_ref = np.asarray(V_ref, dtype=float)
    N = V_ref.shape[0]

    U = generate_uniform_factor_shapes(
        num_nodes=N,
        params=UniformShockShapeParams(rho=rho),
        n_samples=1,
        rng=rng,
    )[0].reshape(-1, 1)

    return xi_from_uniform_base(V_ref, U, lam, scale=scale)

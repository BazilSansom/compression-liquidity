import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from scipy.stats import norm


@dataclass
class ShockShapeParams:
    """
    Parameters for generating *shape* of liquidity shocks via a one-factor
    Gaussian model.

    The idea is to capture *dependence structure* across nodes, without yet
    committing to a particular scaling in £. Scaling relative to VM size or
    buffers can be handled in a separate step.

    Interpretation
    --------------
    - rho in [0, 1] controls how "systemic" the shock is:
        rho = 0  → purely idiosyncratic shocks (independent across nodes)
        rho = 1  → perfectly common shock (all nodes same factor)

    - one_sided:
        If True, we return non-negative "drain factors" (e.g. |Z_i|).
        These can be interpreted as magnitudes of negative liquidity shocks,
        which will later be applied as reductions in buffers.

        If False, we return signed standard-normal factors.

    This class does *not* enforce any scaling; the raw shapes are dimensionless.
    """
    rho: float
    one_sided: bool = True
    mode: Literal["gaussian_factor"] = "gaussian_factor"


def generate_gaussian_factor_shapes(
    num_nodes: int,
    params: ShockShapeParams,
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate correlated shock *shapes* using a one-factor Gaussian model.

    For each sample s and node i, we construct:

        Z_i[s] = sqrt(rho) * Z_common[s] + sqrt(1 - rho) * eps_i[s]

    where Z_common[s] and eps_i[s] are i.i.d. N(0, 1). This yields:

        Var(Z_i)            = 1
        Corr(Z_i, Z_j)      = rho  for i != j

    If params.one_sided is True, we convert these to non-negative "drain
    magnitudes" via abs(Z_i). These can be used as the *shape* of liquidity
    shocks that always reduce available buffers.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N in the network.
    params : ShockShapeParams
        Parameters controlling the dependence structure and one/signed choice.
    n_samples : int, optional
        Number of independent shock vectors to generate. Default: 1.
    seed : int, optional
        Seed for NumPy's Generator, for reproducibility.

    Returns
    -------
    shapes : np.ndarray
        Array of shape (n_samples, num_nodes) containing dimensionless shock
        factors. These have unit marginal variance in the signed case; if
        one_sided=True, they become non-negative magnitudes.

        - If one_sided=False:
            shapes[s, i] ~ approx N(0, 1), Corr(shapes[:,i], shapes[:,j]) ≈ rho
        - If one_sided=True:
            shapes[s, i] = |Z_i|, so non-negative with correlation induced from Z.

    Notes
    -----
    - This function only encodes the *shape* and correlation of shocks.
      Scaling them to £-valued liquidity drains, e.g.

          Δℓ_i = λ * p_i * shape_i

      should be handled in a separate layer that knows about VM size p_i
      or buffer levels.
    """
    rho = float(params.rho)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    rng = np.random.default_rng(seed)

    # Common factor Z_common[s], shared across all nodes for each sample s
    z_common = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 1))

    # Idiosyncratic eps_i[s]
    z_idio = rng.normal(loc=0.0, scale=1.0, size=(n_samples, num_nodes))

    # One-factor Gaussian model
    z = np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * z_idio

    if params.one_sided:
        shapes = np.abs(z)  # non-negative magnitudes representing drains
    else:
        shapes = z          # signed shocks

    return shapes


def gaussian_to_uniform01(z: np.ndarray) -> np.ndarray:
    """
    Map standard normal samples z to uniforms in (0,1) via the normal CDF.
    """
    return norm.cdf(z)



@dataclass
class UniformShockShapeParams:
    """
    Parameters for generating correlated Uniform(0,1) shock shapes using
    the same one-factor Gaussian dependence structure.

    - rho in [0, 1] is interpreted as the correlation parameter in the
      underlying Gaussian factor model. The resulting uniforms are linked
      via a Gaussian copula; their Pearson correlation will be a monotone
      function of rho but not equal to rho.
    """
    rho: float


def generate_uniform_factor_shapes(
    num_nodes: int,
    params: UniformShockShapeParams,
    n_samples: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate correlated Uniform(0,1) shock shapes using the Gaussian factor model.

    Steps:
      1. Generate signed Gaussian factor shapes Z with correlation parameter rho.
      2. Apply the standard normal CDF component-wise to obtain U in (0,1).

    Parameters
    ----------
    num_nodes : int
        Number of nodes N in the network.
    params : UniformShockShapeParams
        Parameters controlling the underlying Gaussian factor correlation.
    n_samples : int, optional
        Number of independent shock vectors to generate. Default: 1.
    seed : int, optional
        Seed for NumPy's Generator, for reproducibility.

    Returns
    -------
    uniforms : np.ndarray
        Array of shape (n_samples, num_nodes) containing Uniform(0,1) shock
        shapes, with dependence given by a Gaussian copula.
    """
    rho = float(params.rho)
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    gauss_params = ShockShapeParams(rho=rho, one_sided=False)
    Z = generate_gaussian_factor_shapes(
        num_nodes=num_nodes,
        params=gauss_params,
        n_samples=n_samples,
        seed=seed,
    )
    U = gaussian_to_uniform01(Z)
    return U

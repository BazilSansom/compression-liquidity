# src/buffers.py
import numpy as np

from src.shocks import xi_from_uniform_base
from src.experiments import find_min_buffers_for_target_shortfall


def behavioural_base_from_V(V: np.ndarray) -> np.ndarray:
    """
    Behavioural buffer *shape* based on outgoing VM obligations.
    """
    V = np.asarray(V, dtype=float)
    return V.sum(axis=1)


def calibrate_buffers_zero_shortfall(
    *,
    V: np.ndarray,
    U_base: np.ndarray,
    lam: float,
    xi_scale: str = "row_sum",
    rel_tol_fpa: float = 1e-8,
) -> np.ndarray:
    """
    Calibrate behavioural buffers to achieve zero aggregate shortfall.

    Parameters
    ----------
    V : np.ndarray
        Liabilities matrix (N,N).
    U_base : np.ndarray
        Shock shape vector (N,1) or (N,).
    lam : float
        Shock intensity.
    xi_scale : {"row_sum","col_sum","total"}
        Scaling rule for xi.
    rel_tol_fpa : float
        Relative tolerance for FPA solver.

    Returns
    -------
    b : np.ndarray
        Buffer vector (N,1) achieving zero aggregate shortfall.
    """
    V = np.asarray(V, dtype=float)
    U = np.asarray(U_base, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, 1)

    xi = xi_from_uniform_base(
        V_ref=V,
        U=U,
        lam=lam,
        scale=xi_scale,
    )

    alpha, b, _ = find_min_buffers_for_target_shortfall(
        V=V,
        xi=xi,
        target_shortfall=0.0,
        buffer_shape_fn=behavioural_base_from_V,
        rel_tol_fpa=rel_tol_fpa,
    )

    return b

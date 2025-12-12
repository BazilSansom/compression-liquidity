from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np

from src.simulation import run_fpa, FPAResult


def _as_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _aggregate_shortfall_for_buffers(
    V: np.ndarray,
    buffers: np.ndarray,
    xi: np.ndarray,
    *,
    rel_tol_fpa: float = 1e-8,
) -> Tuple[float, FPAResult]:
    """
    Helper: compute aggregate shortfall R for a given buffer vector and fixed xi.
    """
    V = np.asarray(V, dtype=float)
    buffers = _as_col(buffers)
    xi = _as_col(xi)

    b0 = buffers - xi
    res = run_fpa(V, b0, rel_tol=rel_tol_fpa, return_paths=False)
    return res.aggregate_shortfall, res


def find_min_buffers_for_target_shortfall(
    V: np.ndarray,
    xi: np.ndarray,
    target_shortfall: float,
    *,
    # exactly one of these two:
    b_base: Optional[np.ndarray] = None,
    buffer_shape_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    # search controls
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
    grow_factor: float = 2.0,
    max_grow_steps: int = 30,
    tol: float = 1e-6,
    max_iter: int = 60,
    rel_tol_fpa: float = 1e-8,
) -> Tuple[float, np.ndarray, FPAResult]:
    """
    Find minimal alpha such that buffers(alpha) = alpha * b_base achieve:

        R(V, buffers(alpha) - xi) <= target_shortfall

    where R is the aggregate shortfall from the FPA.

    Parameters
    ----------
    V : np.ndarray
        N x N VM obligations / liabilities matrix.
    xi : np.ndarray
        N-vector (or N x 1) fixed liquidity drain in £.
        IMPORTANT: xi is treated as exogenous and held fixed during the search.
    target_shortfall : float
        Target aggregate shortfall (often 0.0).
    b_base : np.ndarray, optional
        Base buffer *shape* vector (N,) or (N,1). Used if provided.
    buffer_shape_fn : callable, optional
        Function returning base buffer shape given V (e.g. behavioural rule).
        Used if provided.
    alpha_lo, alpha_hi : float
        Initial bracketing interval for alpha (alpha_lo can be 0).
    grow_factor : float
        How aggressively to expand alpha_hi until feasible.
    max_grow_steps : int
        Maximum number of expansions.
    tol : float
        Relative tolerance for bisection termination.
    max_iter : int
        Maximum bisection iterations.
    rel_tol_fpa : float
        Numerical tolerance for FPA liquidity condition.

    Returns
    -------
    alpha_star : float
        Minimal alpha (to within tolerance) achieving target_shortfall.
    buffers_star : np.ndarray
        N x 1 buffer vector alpha_star * b_base.
    fpa_star : FPAResult
        FPA result at alpha_star.
    """
    V = np.asarray(V, dtype=float)
    xi = _as_col(xi)
    target_shortfall = float(target_shortfall)

    # choose base shape
    if (b_base is None) == (buffer_shape_fn is None):
        raise ValueError("Provide exactly one of b_base or buffer_shape_fn.")

    if buffer_shape_fn is not None:
        b_base = buffer_shape_fn(V)

    b_base = _as_col(b_base)

    # Safety: if b_base is all zeros, scaling does nothing -> infeasible unless target already met.
    if float(np.max(np.abs(b_base))) == 0.0:
        R0, res0 = _aggregate_shortfall_for_buffers(V, buffers=b_base, xi=xi, rel_tol_fpa=rel_tol_fpa)
        if R0 <= target_shortfall:
            return 0.0, b_base, res0
        raise ValueError("b_base is identically zero: cannot reduce shortfall by scaling alpha.")

    def eval_alpha(alpha: float) -> Tuple[float, FPAResult]:
        buffers = alpha * b_base
        R, res = _aggregate_shortfall_for_buffers(V, buffers=buffers, xi=xi, rel_tol_fpa=rel_tol_fpa)
        return R, res

    # 1) Ensure feasibility at alpha_hi by expanding bracket
    lo = float(alpha_lo)
    hi = float(alpha_hi)

    # Optional quick check: if already feasible at lo
    R_lo, res_lo = eval_alpha(lo)
    if R_lo <= target_shortfall:
        buffers_lo = lo * b_base
        return lo, buffers_lo, res_lo

    feasible_res = None
    for _ in range(max_grow_steps):
        R_hi, res_hi = eval_alpha(hi)
        if R_hi <= target_shortfall:
            feasible_res = res_hi
            break
        hi *= grow_factor

    if feasible_res is None:
        raise RuntimeError(
            "Could not bracket a feasible alpha_hi within max_grow_steps. "
            "Try increasing alpha_hi, grow_factor, or max_grow_steps."
        )

    # 2) Bisection
    best_alpha = hi
    best_res = feasible_res

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        R_mid, res_mid = eval_alpha(mid)

        if R_mid <= target_shortfall:
            best_alpha = mid
            best_res = res_mid
            hi = mid
        else:
            lo = mid

        # relative termination (robust when alpha is large)
        if (hi - lo) <= tol * max(1.0, hi):
            break

    buffers_star = best_alpha * b_base
    return best_alpha, buffers_star, best_res

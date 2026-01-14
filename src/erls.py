# src/erls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.shocks import xi_from_uniform_base
from src.buffers import behavioural_base_from_V
from src.experiments import find_min_buffers_for_target_shortfall
from src.experiment_specs import BufferSearchSpec

ZERO_TOL = 1e-12


def _as_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _check_scalable_obligors(*, V: np.ndarray, b_base: np.ndarray, label: str, tol: float = ZERO_TOL) -> None:
    V = np.asarray(V, float)
    b = _as_col(b_base)

    if not np.all(np.isfinite(b)):
        raise ValueError(
            f"[{label}] b_base contains non-finite values: "
            f"n_nonfinite={int(np.sum(~np.isfinite(b)))}"
        )

    oblig = (V.sum(axis=1).reshape(-1, 1) > tol)
    bad = oblig & ~(b > tol)

    if np.any(bad):
        idx = np.where(bad.ravel())[0][:10]
        raise ValueError(
            f"[{label}] Non-scalable obligors: {int(bad.sum())} nodes have row_sum(V)>0 but b_base<=tol. "
            f"Example idx={idx.tolist()}. "
            f"min_b_base={float(np.min(b))}, "
            f"min_pos_b_base={float(np.min(b[b>tol])) if np.any(b>tol) else float('nan')}."
        )


@dataclass
class ERLSResult:
    alpha_pre: float
    alpha_post: float
    kappa: float          # alpha_post / alpha_pre
    B_pre: float
    B_post: float
    erls: float
    # diagnostics
    R_pre: float
    R_post: float


def compute_erls_zero_shortfall(
    V: np.ndarray,
    V_tilde: np.ndarray,
    U_base: np.ndarray,
    lam: float,
    *,
    xi_scale: Literal["row_sum", "col_sum", "total"] = "row_sum",
    search: BufferSearchSpec | None = None,
    rel_tol_fpa: float | None = None,
) -> ERLSResult:
    """
    Behavioural ERLS (behavioural buffers ONLY).

    For a fixed draw (V, U_base) and intensity lam:
      - construct xi using V as the reference scale (xi fixed across compression)
      - pre:  choose base buffer shape b_base_pre = behavioural_base_from_V(V)
      - post: choose base buffer shape b_base_post = behavioural_base_from_V(V_tilde)
      - find minimal scaling alpha_pre, alpha_post so that FPA aggregate shortfall <= target_shortfall

    Notes:
      - feasibility requires scalable obligors: any node with row_sum(V)>0 must have b_base>0
      - search routine enforces nonnegative initial cash via e0 = max(buffers - xi, 0)
    """
    if search is None:
        search = BufferSearchSpec()

    target = float(search.target_shortfall)
    rel_tol = float(search.rel_tol_fpa) if rel_tol_fpa is None else float(rel_tol_fpa)

    V = np.asarray(V, dtype=float)
    V_tilde = np.asarray(V_tilde, dtype=float)

    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("V must be square")
    if V_tilde.shape != V.shape:
        raise ValueError(f"V_tilde must have shape {V.shape}, got {V_tilde.shape}")

    U_base = _as_col(U_base)
    if U_base.shape != (V.shape[0], 1):
        raise ValueError(f"U_base must have shape {(V.shape[0], 1)}, got {U_base.shape}")

    lam = float(lam)
    xi = xi_from_uniform_base(V_ref=V, U=U_base, lam=lam, scale=xi_scale)

    if not np.all(np.isfinite(xi)):
        raise ValueError("xi contains non-finite values")
    if np.any(xi < -1e-15):
        raise ValueError(f"xi has negative entries: min={float(xi.min())}")

    # base shapes
    b_base_pre = _as_col(behavioural_base_from_V(V))
    b_base_post = _as_col(behavioural_base_from_V(V_tilde))

    _check_scalable_obligors(V=V, b_base=b_base_pre, label="pre/behavioural_base")
    _check_scalable_obligors(V=V_tilde, b_base=b_base_post, label="post/behavioural_base")

    search_kwargs = dict(
        alpha_lo=search.alpha_lo,
        alpha_hi=search.alpha_hi,
        grow_factor=search.grow_factor,
        max_grow_steps=search.max_grow_steps,
        tol=search.tol,
        max_iter=search.max_iter,
        rel_tol_fpa=rel_tol,
    )

    alpha_pre, b_pre, fpa_pre = find_min_buffers_for_target_shortfall(
        V=V,
        xi=xi,
        target_shortfall=target,
        b_base=b_base_pre,
        **search_kwargs,
    )

    try:
        alpha_post, b_post, fpa_post = find_min_buffers_for_target_shortfall(
            V=V_tilde,
            xi=xi,
            target_shortfall=target,
            b_base=b_base_post,
            **search_kwargs,
        )
    except RuntimeError as e:
        ob = (V_tilde.sum(axis=1) > ZERO_TOL)
        bb = b_base_post.reshape(-1)
        bb_ob = bb[ob]

        msg = f"{e}\n[diagnostic] lam={lam}, target={target}\n"
        if bb_ob.size:
            smallest = np.argsort(bb_ob)[:10]
            msg += (
                "[diagnostic] b_base_post (obligors): "
                f"min={float(np.min(bb_ob))}, "
                f"p1={float(np.quantile(bb_ob, 0.01))}, "
                f"p5={float(np.quantile(bb_ob, 0.05))}, "
                f"median={float(np.median(bb_ob))}\n"
                f"[diagnostic] smallest obligor indices: {np.where(ob)[0][smallest].tolist()}"
            )
        else:
            msg += f"[diagnostic] No obligors under row_sum(V_tilde)>{ZERO_TOL:g}."
        raise RuntimeError(msg) from e

    B_pre = float(np.sum(b_pre))
    B_post = float(np.sum(b_post))
    erls = 0.0 if B_pre <= 0.0 else (1.0 - (B_post / B_pre))

    alpha_pre_f = float(alpha_pre)
    alpha_post_f = float(alpha_post)
    kappa = float("nan") if alpha_pre_f <= 0.0 else (alpha_post_f / alpha_pre_f)

    return ERLSResult(
        alpha_pre=alpha_pre_f,
        alpha_post=alpha_post_f,
        kappa=float(kappa),
        B_pre=float(B_pre),
        B_post=float(B_post),
        erls=float(erls),
        R_pre=float(fpa_pre.aggregate_shortfall),
        R_post=float(fpa_post.aggregate_shortfall),
    )

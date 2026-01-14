# paper/runners/exp3.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from paper.specs.paper_base import PAPER_MASTER_SEED
from paper.specs.exp3 import Exp3Config  # <-- use the spec-level config
from paper.io import ensure_dir, rows_to_csv, save_json
from paper.meta import make_run_id, utc_now_iso, git_info, python_info, spec_to_jsonable
from paper.paths import ARTIFACT_ROOT

from src.rng import make_streams
from src.experiment_specs import ExperimentSpec
from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import compress
from src.shocks import (
    generate_uniform_factor_shapes,
    UniformShockShapeParams,
    xi_from_uniform_base,
)
from src.buffers import behavioural_base_from_V
from src.simulation import run_fpa
from src.erls import compute_erls_zero_shortfall
from src.stats import flow_hhi







# ============================================================
# Helpers
# ============================================================

def _as_col(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[1] == 1:
        return a
    if a.ndim == 2 and a.shape[0] == 1:
        return a.reshape(-1, 1)
    raise ValueError(f"Expected vector-like; got shape {a.shape}")


def _tiers_from_V(V: np.ndarray, *, eps: float = 1e-15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tiering derived from the *uncompressed* obligations matrix V:
      source: in=0, out>0
      sink:   out=0, in>0
      core:   in>0, out>0
      other:  everything else
    """
    V = np.asarray(V, dtype=float)
    out = V.sum(axis=1)
    inn = V.sum(axis=0)

    source = (inn <= eps) & (out > eps)
    sink = (out <= eps) & (inn > eps)
    core = (out > eps) & (inn > eps)
    other = ~(source | sink | core)
    return source, core, sink, other


def _mean_slack(*, b: np.ndarray, xi_draws: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """
    slack_i = E[ max(b_i - xi_i, 0) ] - ell_i
    """
    b = _as_col(b)[:, 0]
    xi = np.asarray(xi_draws, dtype=float)  # (K,N)
    ell = np.asarray(ell, dtype=float).reshape(-1)

    if xi.ndim != 2:
        raise ValueError(f"xi_draws must be (K,N); got {xi.shape}")
    if xi.shape[1] != b.size or ell.size != b.size:
        raise ValueError(f"Shape mismatch: b {b.shape}, xi {xi.shape}, ell {ell.shape}")

    b_hat = np.maximum(b[None, :] - xi, 0.0)
    return b_hat.mean(axis=0) - ell


def _estimate_vulnerability(
    *,
    V: np.ndarray,
    b: np.ndarray,
    xi_draws: np.ndarray,
    rel_tol_fpa: float,
    fail_tol: float,
) -> np.ndarray:
    """
    nu_i = P[ residual_obligations_i > 0 ] estimated by Monte Carlo over xi_draws.
    Returns (N,)
    """
    V = np.asarray(V, dtype=float)
    b = _as_col(b)  # (N,1)
    xi = np.asarray(xi_draws, dtype=float)  # (K,N)

    N = V.shape[0]
    if V.shape != (N, N):
        raise ValueError(f"V must be (N,N); got {V.shape}")
    if b.shape != (N, 1):
        raise ValueError(f"b must be (N,1); got {b.shape}")
    if xi.ndim != 2 or xi.shape[1] != N:
        raise ValueError(f"xi_draws must be (K,N); got {xi.shape}")

    K = xi.shape[0]
    counts = np.zeros(N, dtype=float)

    for k in range(K):
        bhat = np.maximum(b[:, 0] - xi[k, :], 0.0).reshape(-1, 1)
        res = run_fpa(V, bhat, rel_tol=rel_tol_fpa, return_paths=False, validate=False)
        resid = np.asarray(res.residual_obligations, dtype=float).reshape(-1)
        counts += (resid > fail_tol).astype(float)

    return counts / float(K)


def _gap_stats(
    q_bff: np.ndarray,
    q_maxc: np.ndarray,
    *,
    mask: Optional[np.ndarray],
    topk: int,
) -> Dict[str, float]:
    """
    Basic vulnerability gap summaries over nodes selected by mask:
      diff = q_maxc - q_bff
    """
    qb = np.asarray(q_bff, dtype=float).reshape(-1)
    qm = np.asarray(q_maxc, dtype=float).reshape(-1)
    if qb.shape != qm.shape:
        raise ValueError(f"q shapes differ: {qb.shape} vs {qm.shape}")

    if mask is None:
        mask = np.ones_like(qb, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)

    qb = qb[mask]
    qm = qm[mask]
    if qb.size == 0:
        return {
            "n_nodes": 0.0,
            "mean_gap": float("nan"),
            "dom_share": float("nan"),
            "topk_gap": float("nan"),
            "mean_bff": float("nan"),
            "mean_maxc": float("nan"),
        }

    diff = qm - qb
    k = min(int(topk), qb.size)

    return {
        "n_nodes": float(qb.size),
        "mean_gap": float(diff.mean()),
        "dom_share": float((diff > 0.0).mean()),
        "topk_gap": float(np.sort(qm)[-k:].sum() - np.sort(qb)[-k:].sum()),
        "mean_bff": float(qb.mean()),
        "mean_maxc": float(qm.mean()),
    }


# ============================================================
# Calibration: alpha_ref from repeated root-search over U shapes
# ============================================================

def calibrate_alpha_ref(
    *,
    V_unc: np.ndarray,
    V_tilde: np.ndarray,
    U_cals: np.ndarray,      # (n_calib, N) or list-like; shared across methods for CRN
    lam_ref: float,
    xi_scale: float,
    search,
    agg: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute alpha_ref by running compute_erls_zero_shortfall on many calibration shapes U_cals.
    Uses the post-compression alpha_post series; aggregates by median/mean.

    Returns: (alpha_ref, diag)
    """
    alpha_pre = []
    alpha_post = []

    for U in U_cals:
        U_base = np.asarray(U).reshape(-1, 1)
        res = compute_erls_zero_shortfall(
            V=V_unc,
            V_tilde=V_tilde,
            U_base=U_base,
            lam=float(lam_ref),
            xi_scale=xi_scale,
            search=search,
        )
        alpha_pre.append(float(res.alpha_pre))
        alpha_post.append(float(res.alpha_post))

    a = np.asarray(alpha_post, dtype=float)

    if agg == "median":
        alpha_ref = float(np.nanmedian(a))
    elif agg == "mean":
        alpha_ref = float(np.nanmean(a))
    else:
        raise ValueError(f"Unknown agg='{agg}' (use 'median' or 'mean')")

    diag = {
        "alpha_pre_median": float(np.nanmedian(alpha_pre)),
        "alpha_post_median": float(np.nanmedian(alpha_post)),
        "alpha_post_q25": float(np.nanquantile(a, 0.25)),
        "alpha_post_q75": float(np.nanquantile(a, 0.75)),
        "n_calib_actual": float(len(alpha_post)),
    }
    return alpha_ref, diag


# ============================================================
# Evaluation: vulnerability + slack under common alpha
# ============================================================

def eval_common_alpha(
    *,
    V_unc: np.ndarray,
    V_bff: np.ndarray,
    V_maxc: np.ndarray,
    alpha_common: float,      # = alpha_ref_bff
    lam_eval: float,
    xi_scale: float,
    rho_xi: float,
    rel_tol_fpa: float,
    rng_eval,
    n_xi_draws: int,
    fail_tol: float,
) -> Dict[str, Any]:
    N = V_unc.shape[0]

    # planned buffers at common alpha (method-specific bases)
    b_bff = _as_col(alpha_common * behavioural_base_from_V(V_bff))
    b_maxc = _as_col(alpha_common * behavioural_base_from_V(V_maxc))

    # draw evaluation shapes, shared across methods
    U_draws = generate_uniform_factor_shapes(
        num_nodes=N,
        params=UniformShockShapeParams(rho=rho_xi),
        n_samples=int(n_xi_draws),
        rng=rng_eval,
    )

    xi_draws = np.empty((int(n_xi_draws), N), dtype=float)
    for k, U in enumerate(U_draws):
        xi_k = xi_from_uniform_base(
            V_ref=V_unc,  # common scaling anchor across methods
            U=np.asarray(U).reshape(-1, 1),
            lam=float(lam_eval),
            scale=xi_scale,
        )
        xi_draws[k, :] = np.asarray(xi_k, dtype=float).reshape(-1)

    # vulnerabilities
    q_bff = _estimate_vulnerability(
        V=V_bff,
        b=b_bff,
        xi_draws=xi_draws,
        rel_tol_fpa=rel_tol_fpa,
        fail_tol=fail_tol,
    )
    q_maxc = _estimate_vulnerability(
        V=V_maxc,
        b=b_maxc,
        xi_draws=xi_draws,
        rel_tol_fpa=rel_tol_fpa,
        fail_tol=fail_tol,
    )

    # slacks
    ell_bff = V_bff.sum(axis=1)
    ell_maxc = V_maxc.sum(axis=1)

    slack_bff = _mean_slack(b=b_bff, xi_draws=xi_draws, ell=ell_bff)
    slack_maxc = _mean_slack(b=b_maxc, xi_draws=xi_draws, ell=ell_maxc)

    return {
        "alpha_common": float(alpha_common),
        "b_bff": b_bff,
        "b_maxc": b_maxc,
        "xi_draws": xi_draws,
        "q_bff": q_bff,
        "q_maxc": q_maxc,
        "delta_vuln_common": q_maxc - q_bff,
        "slack_bff": slack_bff,
        "slack_maxc": slack_maxc,
        "delta_slack_common": slack_maxc - slack_bff,
    }


# ============================================================
# One draw: network -> compress -> calibrate -> eval at common alpha
# ============================================================

def run_one_draw_common_alpha(draw_id: int, config: Exp3Config) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = config.spec
    draw_id = int(draw_id)

    streams = make_streams(master_seed=PAPER_MASTER_SEED, draw_id=draw_id)

    # --- 1) Network draw ---
    G = generate_three_tier_network(
        n_core=spec.network.n_core,
        n_source=spec.network.n_source,
        n_sink=spec.network.n_sink,
        p=spec.network.p,
        weight_mode=spec.network.weight_mode,
        alpha_weights=spec.network.alpha_weights,
        scale_weights=spec.network.scale_weights,
        rng_topology=streams.net_topology,
        rng_weights=streams.net_weights,
        degree_mode=spec.network.degree_mode,
        round_to=spec.network.round_to,
    )
    if spec.network.use_lcc:
        G = extract_largest_component(G)

    V_unc = np.asarray(G.W, dtype=float)
    N = int(V_unc.shape[0])
    if V_unc.shape != (N, N):
        raise ValueError(f"V_unc must be (N,N); got {V_unc.shape}")

    # tiers from uncompressed
    source_mask, core_mask, sink_mask, other_mask = _tiers_from_V(V_unc)
    tier = np.full(N, "other", dtype=object)
    tier[source_mask] = "source"
    tier[core_mask] = "core"
    tier[sink_mask] = "sink"

    # --- 2) Compress ---
    comp_bff = compress(
        G,
        method="bff",
        solver=spec.compression.solver,
        tol_zero=spec.compression.tol_zero,
        require_conservative=spec.compression.require_conservative,
        require_full_conservative=spec.compression.require_full_conservative,
    )
    V_bff = np.asarray(comp_bff.compressed.W, dtype=float)

    comp_maxc = compress(
        G,
        method="maxc",
        solver=config.maxc_solver,
        tol_zero=spec.compression.tol_zero,
        require_conservative=spec.compression.require_conservative,
        require_full_conservative=spec.compression.require_full_conservative,
    )
    V_maxc = np.asarray(comp_maxc.compressed.W, dtype=float)

    if V_bff.shape != (N, N) or V_maxc.shape != (N, N):
        raise ValueError(f"Compressed shapes mismatch: bff {V_bff.shape}, maxc {V_maxc.shape}, N={N}")

    comp_ratio_bff = float(comp_bff.gross_after) / float(comp_bff.gross_before) if comp_bff.gross_before > 0 else float("nan")
    comp_ratio_maxc = float(comp_maxc.gross_after) / float(comp_maxc.gross_before) if comp_maxc.gross_before > 0 else float("nan")

    # --- 3) Calibration shapes (shared CRN across methods) ---
    U_cals = generate_uniform_factor_shapes(
        num_nodes=N,
        params=UniformShockShapeParams(rho=spec.shock.rho_xi),
        n_samples=int(config.n_calib),
        rng=streams.shock_cal,  # deterministic per draw_id via make_streams
    )

    # --- 4) Calibrate alpha_ref for each method on SAME U_cals ---
    alpha_ref_bff, diag_bff = calibrate_alpha_ref(
        V_unc=V_unc,
        V_tilde=V_bff,
        U_cals=U_cals,
        lam_ref=float(config.lam_ref),
        xi_scale=spec.shock.xi_scale,
        search=spec.search,
        agg=config.alpha_agg,
    )
    alpha_ref_maxc, diag_maxc = calibrate_alpha_ref(
        V_unc=V_unc,
        V_tilde=V_maxc,
        U_cals=U_cals,
        lam_ref=float(config.lam_ref),
        xi_scale=spec.shock.xi_scale,
        search=spec.search,
        agg=config.alpha_agg,
    )

    # sanity check: alpha_pre_median should match across methods (same V_unc, same U_cals)
    tol = 1e-10 * max(1.0, abs(diag_bff["alpha_pre_median"]))
    if abs(diag_bff["alpha_pre_median"] - diag_maxc["alpha_pre_median"]) > tol:
        raise AssertionError(
            f"alpha_pre_median mismatch: bff={diag_bff['alpha_pre_median']:.12g}, "
            f"maxc={diag_maxc['alpha_pre_median']:.12g}"
        )

    # --- 5) Evaluate at common alpha = alpha_ref_bff ---
    rng_eval = getattr(streams, "shock_mc", streams.shock_shape)
    eval_out = eval_common_alpha(
        V_unc=V_unc,
        V_bff=V_bff,
        V_maxc=V_maxc,
        alpha_common=float(alpha_ref_bff),
        lam_eval=float(config.lam_eval),
        xi_scale=spec.shock.xi_scale,
        rho_xi=spec.shock.rho_xi,
        rel_tol_fpa=spec.search.rel_tol_fpa,
        rng_eval=rng_eval,
        n_xi_draws=int(config.n_xi_draws),
        fail_tol=float(config.fail_tol),
    )

    q_bff = eval_out["q_bff"]
    q_maxc = eval_out["q_maxc"]
    delta_v = eval_out["delta_vuln_common"]
    slack_bff = eval_out["slack_bff"]
    slack_maxc = eval_out["slack_maxc"]
    delta_s = eval_out["delta_slack_common"]

    # controls
    in_unc = V_unc.sum(axis=0)
    in_bff = V_bff.sum(axis=0)
    in_maxc = V_maxc.sum(axis=0)

    ell_bff = V_bff.sum(axis=1)
    ell_maxc = V_maxc.sum(axis=1)

    hhi_out_bff = flow_hhi(V_bff, mode="outgoing")
    hhi_out_maxc = flow_hhi(V_maxc, mode="outgoing")


    # network-level summaries
    stats_all = _gap_stats(q_bff, q_maxc, mask=None, topk=config.topk)
    stats_core = _gap_stats(q_bff, q_maxc, mask=core_mask, topk=config.topk)

    network_row: Dict[str, Any] = {
        "draw_id": draw_id,
        "N": int(N),
        "lam_ref": float(config.lam_ref),
        "lam_eval": float(config.lam_eval),
        "n_calib": int(config.n_calib),
        "alpha_agg": str(config.alpha_agg),
        "n_xi_draws": int(config.n_xi_draws),

        "alpha_ref_bff": float(alpha_ref_bff),
        "alpha_ref_maxc": float(alpha_ref_maxc),
        "alpha_pre_median": float(diag_bff["alpha_pre_median"]),
        "alpha_bff_q25": float(diag_bff["alpha_post_q25"]),
        "alpha_bff_q75": float(diag_bff["alpha_post_q75"]),
        "alpha_maxc_q25": float(diag_maxc["alpha_post_q25"]),
        "alpha_maxc_q75": float(diag_maxc["alpha_post_q75"]),

        "comp_ratio_bff": float(comp_ratio_bff),
        "comp_ratio_maxc": float(comp_ratio_maxc),

        # all nodes
        "all_n": stats_all["n_nodes"],
        "all_mean_gap": stats_all["mean_gap"],
        "all_dom_share": stats_all["dom_share"],
        "all_topk_gap": stats_all["topk_gap"],
        "all_mean_bff": stats_all["mean_bff"],
        "all_mean_maxc": stats_all["mean_maxc"],

        # core nodes
        "core_n": stats_core["n_nodes"],
        "core_mean_gap": stats_core["mean_gap"],
        "core_dom_share": stats_core["dom_share"],
        "core_topk_gap": stats_core["topk_gap"],
        "core_mean_bff": stats_core["mean_bff"],
        "core_mean_maxc": stats_core["mean_maxc"],
    }

    # node panel rows (for regression / scatter)
    node_rows: List[Dict[str, Any]] = []
    for i in range(N):
        ell_bff_i = float(ell_bff[i])
        node_rows.append(
            {
                "net_id": draw_id,
                "node": int(i),
                "tier": str(tier[i]),

                "lam_ref": float(config.lam_ref),
                "lam_eval": float(config.lam_eval),
                "alpha_common": float(alpha_ref_bff),

                "vuln_bff_common": float(q_bff[i]),
                "vuln_maxc_common": float(q_maxc[i]),
                "delta_vuln_common": float(delta_v[i]),

                "slack_bff_common": float(slack_bff[i]),
                "slack_maxc_common": float(slack_maxc[i]),
                "delta_slack_common": float(delta_s[i]),

                "ell_bff": float(ell_bff[i]),
                "ell_maxc": float(ell_maxc[i]),

                "in_unc": float(in_unc[i]),
                "in_bff": float(in_bff[i]),
                "in_maxc": float(in_maxc[i]),
                "incoming_reliance_bff": float(in_bff[i]) / (ell_bff_i + 1e-12),

                "hhi_out_bff": float(hhi_out_bff[i]),
                "hhi_out_maxc": float(hhi_out_maxc[i]),
                "delta_hhi_out": float(hhi_out_maxc[i] - hhi_out_bff[i]),

                "comp_ratio_bff": float(comp_ratio_bff),
                "comp_ratio_maxc": float(comp_ratio_maxc),
            }
        )

    if draw_id % 10 == 0:
        print(
            f"[draw {draw_id:4d}] core_mean_gap={stats_core['mean_gap']:+.4f}  "
            f"core_dom_share={stats_core['dom_share']:.3f}  "
            f"alpha_ref(bff,maxc)=({alpha_ref_bff:.3g},{alpha_ref_maxc:.3g})"
        )

    return network_row, node_rows


# ============================================================
# Multiple draws: thin wrapper (no IO here yet)
# ============================================================

def run_many_draws_common_alpha(draw_ids: Sequence[int], config: Exp3Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    network_rows: List[Dict[str, Any]] = []
    node_rows: List[Dict[str, Any]] = []
    for d in draw_ids:
        net_row, nodes = run_one_draw_common_alpha(int(d), config)
        network_rows.append(net_row)
        node_rows.extend(nodes)
    return network_rows, node_rows

# ============================================================
# Main entry point: run_exp3
# ============================================================

@dataclass(frozen=True)
class Exp3RunResult:
    run_id: str
    artifact_dir: Path
    network_summary_csv: Path
    node_panel_csv: Path
    meta_json: Path
    spec_json: Path


def run_exp3(
    config: Exp3Config,
    draw_ids: Sequence[int],
    *,
    tag: str = "paper",
) -> Exp3RunResult:
    run_id = make_run_id("exp3", tag)
    artifact_dir = ARTIFACT_ROOT / "exp3" / run_id
    ensure_dir(artifact_dir)

    network_rows, node_rows = run_many_draws_common_alpha(draw_ids, config)

    network_summary_csv = artifact_dir / "network_summary.csv"
    node_panel_csv = artifact_dir / "node_panel.csv"
    meta_json = artifact_dir / "meta.json"
    spec_json = artifact_dir / "spec.json"

    rows_to_csv(network_rows, network_summary_csv)
    rows_to_csv(node_rows, node_panel_csv)

    save_json(
        spec_json,
        {
            "config": {
                "maxc_solver": config.maxc_solver,
                "lam_ref": float(config.lam_ref),
                "lam_eval": float(config.lam_eval),
                "n_calib": int(config.n_calib),
                "alpha_agg": str(config.alpha_agg),
                "n_xi_draws": int(config.n_xi_draws),
                "topk": int(config.topk),
                "fail_tol": float(config.fail_tol),
            },
            "experiment_spec": spec_to_jsonable(config.spec),
        },
    )

    save_json(
        meta_json,
        {
            "run_id": run_id,
            "experiment": "exp3",
            "tag": tag,
            "timestamp_utc": utc_now_iso(),
            "git": git_info(),
            "python": python_info(),
            "master_seed": int(PAPER_MASTER_SEED),
            "draw_ids": list(map(int, draw_ids)),
            "artifact_dir": str(artifact_dir),
        },
    )

    return Exp3RunResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        network_summary_csv=network_summary_csv,
        node_panel_csv=node_panel_csv,
        meta_json=meta_json,
        spec_json=spec_json,
    )

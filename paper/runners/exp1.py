# paper/runners/exp1.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from paper.io import ensure_dir, save_json, rows_to_csv
from paper.meta import make_run_id, utc_now_iso, git_info, python_info, spec_to_jsonable

from paper.paths import ARTIFACT_ROOT
from paper.specs.exp1 import Exp1Config
from paper.specs.paper_base import PAPER_MASTER_SEED

from src.experiment_specs import ExperimentSpec
from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import compress
from src.rng import make_streams
from src.shocks import (
    generate_uniform_factor_shapes,
    UniformShockShapeParams,
    xi_from_uniform_base,
)
from src.buffers import behavioural_base_from_V
from src.simulation import run_fpa


# ----------------------------
# Exp1 mechanism decomposition
# ----------------------------

def mechanism_decomp_4way(fpa_pre, fpa_post) -> dict[str, float | int]:
    act_pre = np.asarray(fpa_pre.activation_round, dtype=int).reshape(-1)
    act_post = np.asarray(fpa_post.activation_round, dtype=int).reshape(-1)

    # IMPORTANT: "paid" means activation_round >= 1 (not >= 0),
    # because 0 means "zero liabilities at t=0" in your coding.
    active_pre = act_pre >= 1
    active_post = act_post >= 1

    N = act_pre.size
    n_active_pre = int(active_pre.sum())
    n_active_post = int(active_post.sum())

    new_active = active_post & (~active_pre)
    still_inactive = ~active_post

     #direct_new = new_active & (act_post <= 1)
    direct_new = new_active & (act_post == 1)
    indirect_new = new_active & (act_post >= 2)

    # Use steady-state l and b to form shortfall consistently
    l_pre = np.asarray(fpa_pre.residual_obligations, dtype=float).reshape(-1)
    l_post = np.asarray(fpa_post.residual_obligations, dtype=float).reshape(-1)
    b_pre = np.asarray(fpa_pre.final_cash, dtype=float).reshape(-1)
    b_post = np.asarray(fpa_post.final_cash, dtype=float).reshape(-1)

    sf_pre = np.maximum(0.0, l_pre - b_pre)
    sf_post = np.maximum(0.0, l_post - b_post)

    # New actives: exact realised reduction for these nodes
    dR_new_trigger = float((sf_pre[direct_new] - sf_post[direct_new]).sum())
    dR_new_late = float((sf_pre[indirect_new] - sf_post[indirect_new]).sum())
    dR_new_active = dR_new_trigger + dR_new_late

    # Still inactive: exact split into cash vs obligation effects
    idx = np.where(still_inactive)[0]
    if idx.size > 0:
        # Step 1: change cash b_pre -> b_post holding l at l_pre
        cash_relief = np.maximum(0.0, l_pre[idx] - b_pre[idx]) - np.maximum(0.0, l_pre[idx] - b_post[idx])

        # Step 2: change obligations l_pre -> l_post holding cash at b_post
        oblig_relief = np.maximum(0.0, l_pre[idx] - b_post[idx]) - np.maximum(0.0, l_post[idx] - b_post[idx])

        dR_still_inactive_cash = float(cash_relief.sum())
        dR_still_inactive_oblig = float(oblig_relief.sum())
    else:
        dR_still_inactive_cash = 0.0
        dR_still_inactive_oblig = 0.0

    n_new = int(new_active.sum())
    n_new_t1 = int(direct_new.sum())
    n_new_late = int(indirect_new.sum())

    return {
        "n_newly_active": n_new,
        "n_newly_active_t1": n_new_t1,
        "n_newly_active_late": n_new_late,
        "share_newly_active_t1": (n_new_t1 / n_new) if n_new > 0 else float("nan"),

        "n_active_pre": n_active_pre,
        "n_active_post": n_active_post,
        "share_active_pre": n_active_pre / N,
        "share_active_post": n_active_post / N,

        "delta_R_new_trigger": dR_new_trigger,
        "delta_R_new_late": dR_new_late,
        "delta_R_new_active": dR_new_active,

        "delta_R_still_inactive_cash_relief": dR_still_inactive_cash,
        "delta_R_still_inactive_obligation_relief": dR_still_inactive_oblig,
    }


# ----------------------------
# Core compute
# ----------------------------

def run_exp1_method(
    spec: ExperimentSpec,
    *,
    draw_ids: Sequence[int],
    buffer_theta: float,
    compression_method: str,
    compression_solver: str | None = None,
    #use_lcc: bool = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    method = compression_method
    solver = compression_solver or spec.compression.solver

    #net_seed_base = spec.network.seed_offset
    #shock_seed_base = spec.shock.seed_offset
    
    for draw_id in draw_ids:
        streams = make_streams(master_seed=PAPER_MASTER_SEED, draw_id=int(draw_id))

        # --- 1) draw network ---
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
        
        #if use_lcc:
        if spec.network.use_lcc:
            G = extract_largest_component(G)
        
        V = np.asarray(G.W, dtype=float)
        N = int(G.num_nodes)


        # --- 2) compress ---
        comp = compress(
            G,
            method=method,
            solver=solver,
            tol_zero=spec.compression.tol_zero,
            require_conservative=spec.compression.require_conservative,
            require_full_conservative=spec.compression.require_full_conservative,
        )
        V_tilde = np.asarray(comp.compressed.W, dtype=float)

        # --- 3) fixed buffers ---
        b_base = behavioural_base_from_V(V)
        b_base = np.asarray(b_base, dtype=float).reshape(-1, 1)
        b_fixed = float(buffer_theta) * b_base

        # --- 4) base shock shape (fixed across lambda) ---
        U_base = generate_uniform_factor_shapes(
            num_nodes=N,
            params=UniformShockShapeParams(rho=spec.shock.rho_xi),
            n_samples=1,
            rng=streams.shock_shape,
        )[0].reshape(-1, 1)

        # --- 5) sweep lambda ---
        for lam in spec.shock.lam_grid:
            xi = xi_from_uniform_base(V_ref=V, U=U_base, lam=float(lam), scale=spec.shock.xi_scale)
            b0 = b_fixed - xi

            fpa_pre = run_fpa(V, b0, rel_tol=spec.search.rel_tol_fpa, return_paths=False)
            fpa_post = run_fpa(V_tilde, b0, rel_tol=spec.search.rel_tol_fpa, return_paths=False)

            R_pre = float(fpa_pre.aggregate_shortfall)
            R_post = float(fpa_post.aggregate_shortfall)
            delta_R_raw = R_pre - R_post

            # clamp tiny numerical negatives (stored variable only)
            delta_R = delta_R_raw
            eps_R = 1e-12 * max(1.0, abs(R_pre), abs(R_post))
            if delta_R < 0.0 and abs(delta_R) < eps_R:
                delta_R = 0.0

            gross_before = float(comp.gross_before)
            delta_R_norm = delta_R / gross_before if gross_before > 0 else float("nan")

            #mech = mechanism_decomp_4way(V_tilde, fpa_pre, fpa_post)
            mech = mechanism_decomp_4way(fpa_pre, fpa_post)

            # identity check uses raw ΔR
            lhs = delta_R_raw
            rhs = (
                mech["delta_R_new_active"]
                + mech["delta_R_still_inactive_cash_relief"]
                + mech["delta_R_still_inactive_obligation_relief"]
            )
            tol = 1e-9 * max(1.0, abs(lhs), abs(rhs))
            if abs(lhs - rhs) > tol:
                raise AssertionError(f"ΔR decomposition failed: lhs={lhs:.6g}, rhs={rhs:.6g}")
            
            if abs(delta_R) > 0:
                share_new_active = mech["delta_R_new_active"] / delta_R
                share_cash = mech["delta_R_still_inactive_cash_relief"] / delta_R
                share_oblig = mech["delta_R_still_inactive_obligation_relief"] / delta_R
            else:
                share_new_active = float("nan")
                share_cash = float("nan")
                share_oblig = float("nan")

            rows.append(
                {
                    "spec": spec.name,
                    "draw_id": int(draw_id),
                    "N": int(N),
                    "use_lcc": bool(spec.network.use_lcc),
                    "compression_method": str(method),
                    "compression_solver": str(solver),
                    "buffer_theta": float(buffer_theta),
                    "rho_xi": float(spec.shock.rho_xi),
                    "xi_scale": spec.shock.xi_scale,
                    "lam": float(lam),

                    "R_pre": R_pre,
                    "R_post": R_post,
                    "delta_R": float(delta_R),
                    "delta_R_raw": float(delta_R_raw),
                    "delta_R_norm": float(delta_R_norm),

                    "gross_before": float(comp.gross_before),
                    "gross_after": float(comp.gross_after),
                    "savings_frac": float(comp.savings_frac),

                    # Activation / mechanism
                    "n_newly_active": mech["n_newly_active"],
                    "n_newly_active_t1": mech["n_newly_active_t1"],
                    "n_newly_active_late": mech["n_newly_active_late"],
                    "share_newly_active_t1": mech["share_newly_active_t1"],

                    "n_active_pre": mech["n_active_pre"],
                    "n_active_post": mech["n_active_post"],
                    "share_active_pre": mech["share_active_pre"],
                    "share_active_post": mech["share_active_post"],

                    "delta_R_new_trigger": mech["delta_R_new_trigger"],
                    "delta_R_new_late": mech["delta_R_new_late"],
                    "delta_R_new_active": mech["delta_R_new_active"],

                    "delta_R_still_inactive_cash_relief": mech["delta_R_still_inactive_cash_relief"],
                    "delta_R_still_inactive_obligation_relief": mech["delta_R_still_inactive_obligation_relief"],

                    "share_new_active": share_new_active,
                    "share_still_inactive_cash_relief": share_cash,
                    "share_still_inactive_obligation_relief": share_oblig,
                }
            )

    return rows


def _summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    # Summary for plotting: per method x lam
    def q(p: float):
        return lambda s: float(np.nanquantile(s.to_numpy(), p))

    grp = draws.groupby(["compression_method", "lam"], as_index=False)

    out = grp.agg(
        n=("delta_R_norm", "size"),
        delta_R_norm_mean=("delta_R_norm", "mean"),
        delta_R_norm_p25=("delta_R_norm", q(0.25)),
        delta_R_norm_p50=("delta_R_norm", q(0.50)),
        delta_R_norm_p75=("delta_R_norm", q(0.75)),
        share_new_active_mean=("share_new_active", "mean"),
        share_cash_mean=("share_still_inactive_cash_relief", "mean"),
        share_oblig_mean=("share_still_inactive_obligation_relief", "mean"),
        share_active_post_mean=("share_active_post", "mean"),
    )
    return out.sort_values(["compression_method", "lam"]).reset_index(drop=True)


def _make_arrays_npz(summary: pd.DataFrame, out_path: Path) -> None:
    methods = sorted(summary["compression_method"].unique().tolist())
    lam_grid = np.array(sorted(summary["lam"].unique().tolist()), dtype=float)

    def pivot(col: str) -> np.ndarray:
        p = summary.pivot(index="compression_method", columns="lam", values=col).reindex(methods)
        return p.to_numpy(dtype=float)

    np.savez(
        out_path,
        methods=np.array(methods, dtype=object),
        lam_grid=lam_grid,
        delta_R_norm_mean=pivot("delta_R_norm_mean"),
        delta_R_norm_p25=pivot("delta_R_norm_p25"),
        delta_R_norm_p50=pivot("delta_R_norm_p50"),
        delta_R_norm_p75=pivot("delta_R_norm_p75"),
    )


@dataclass(frozen=True)
class Exp1RunResult:
    run_id: str
    artifact_dir: Path
    draws_csv: Path
    summary_csv: Path
    arrays_npz: Path
    meta_json: Path
    spec_json: Path


def run_exp1(config: Exp1Config, draw_ids: Sequence[int], tag: str = "main") -> Exp1RunResult:
    run_id = make_run_id("exp1", tag)
    artifact_dir = ARTIFACT_ROOT / "exp1" / run_id
    ensure_dir(artifact_dir)

    # run methods
    all_rows: List[Dict[str, Any]] = []
    for m in config.methods:
        solver = config.maxc_solver if m == "maxc" else None
        rows = run_exp1_method(
            config.spec,
            draw_ids=draw_ids,
            buffer_theta=config.buffer_theta,
            compression_method=m,
            compression_solver=solver,
            #use_lcc=config.use_lcc,
        )
        all_rows.extend(rows)

    draws_df = pd.DataFrame(all_rows)
    summary_df = _summarize_draws(draws_df)

    draws_csv = artifact_dir / "draws.csv"
    summary_csv = artifact_dir / "summary.csv"
    arrays_npz = artifact_dir / "arrays.npz"
    meta_json = artifact_dir / "meta.json"
    spec_json = artifact_dir / "spec.json"

    rows_to_csv(all_rows, draws_csv)
    summary_df.to_csv(summary_csv, index=False)
    _make_arrays_npz(summary_df, arrays_npz)

    save_json(
        spec_json,
        {
            "config": {
                "buffer_theta": config.buffer_theta,
                "methods": list(config.methods),
                "maxc_solver": config.maxc_solver,
                #"use_lcc": config.use_lcc,
                "use_lcc": bool(config.spec.network.use_lcc)

            },
            "experiment_spec": spec_to_jsonable(config.spec),
        },
    )

    save_json(
        meta_json,
        {
            "run_id": run_id,
            "experiment": "exp1",
            "tag": tag,
            "timestamp_utc": utc_now_iso(),
            "git": git_info(),
            "python": python_info(),
            "master_seed": int(PAPER_MASTER_SEED),
            "draw_ids": list(map(int, draw_ids)),
            "artifact_dir": str(artifact_dir),
        },
    )

    return Exp1RunResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        draws_csv=draws_csv,
        summary_csv=summary_csv,
        arrays_npz=arrays_npz,
        meta_json=meta_json,
        spec_json=spec_json,
    )

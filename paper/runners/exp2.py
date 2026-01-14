from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from paper.io import ensure_dir, save_json, rows_to_csv
from paper.meta import make_run_id, utc_now_iso, git_info, python_info, spec_to_jsonable

from paper.paths import ARTIFACT_ROOT
from paper.specs.exp2 import Exp2Config
from paper.specs.paper_base import PAPER_MASTER_SEED

from src.rng import make_streams
from src.experiment_specs import ExperimentSpec
from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import compress
from src.shocks import generate_uniform_factor_shapes, UniformShockShapeParams
from src.erls import compute_erls_zero_shortfall



def run_exp2_method(
    spec: ExperimentSpec,
    *,
    draw_ids: Sequence[int],
    compression_method: str,
    compression_solver: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Run Exp2 (ERLS vs shock intensity λ) for a single compression method.

    For each `draw_id`:
      1) draw a network (topology + weights) using dedicated RNG streams,
      2) compute a compressed network under the chosen compression method,
      3) draw a single base shock direction `U_base` (fixed across λ for that draw),
      4) sweep λ over `spec.shock.lam_grid` and compute ERLS / κ / α-thresholds.

    The output is a list of per-(draw_id, λ) rows suitable for CSV export and
    downstream aggregation/plotting.
    """

    rows: List[Dict[str, Any]] = []

    method = compression_method
    solver = compression_solver or spec.compression.solver

    for draw_id in draw_ids:
        draw_id = int(draw_id)
        
        streams = make_streams(master_seed=PAPER_MASTER_SEED, draw_id=draw_id)
        
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

        gross_before = float(comp.gross_before)
        gross_after = float(comp.gross_after)
        gross_ratio = (gross_after / gross_before) if gross_before > 0 else float("nan")

        # --- 3) draw base shock U_base once (fixed across lambda) ---
        U_base = generate_uniform_factor_shapes(
            num_nodes=N,
            params=UniformShockShapeParams(rho=spec.shock.rho_xi),
            n_samples=1,
            rng=streams.shock_shape,
        )[0].reshape(-1, 1)

        # --- 4) sweep lambda ---

        #  validate shapes 
        if V.shape != (N, N):
            raise ValueError(f"V must be (N,N) with N={N}; got {V.shape}")
        if V_tilde.shape != (N, N):
            raise ValueError(f"V_tilde must be (N,N) with N={N}; got {V_tilde.shape}")
        if U_base.shape != (N, 1):
            raise ValueError(f"U_base must be (N,1) with N={N}; got {U_base.shape}")

        for lam in spec.shock.lam_grid:
            res = compute_erls_zero_shortfall(
                V=V,
                V_tilde=V_tilde,
                U_base=U_base,
                lam=float(lam),
                xi_scale=spec.shock.xi_scale,
                search=spec.search,
            )

            alpha_pre = float(res.alpha_pre)
            alpha_post = float(res.alpha_post)
            kappa = float(res.kappa)

            erls_from_parts = (
                float(1.0 - kappa * gross_ratio)
                if np.isfinite(kappa) and np.isfinite(gross_ratio)
                else float("nan")
            )

            rows.append(
                {
                    "spec": spec.name,
                    "draw_id": draw_id,
                    "N": N,
                    "use_lcc": bool(spec.network.use_lcc),
                    "compression_method": str(method),
                    "compression_solver": str(solver),
                    "rho_xi": float(spec.shock.rho_xi),
                    "xi_scale": spec.shock.xi_scale,
                    "lam": float(lam),
                    "alpha_pre": alpha_pre,
                    "alpha_post": alpha_post,
                    "kappa": kappa,
                    "gross_before": gross_before,
                    "gross_after": gross_after,
                    "gross_ratio": float(gross_ratio),
                    "savings_frac": float(comp.savings_frac),
                    "erls": float(res.erls),
                    "erls_from_parts": float(erls_from_parts),
                    "R_pre": float(res.R_pre),
                    "R_post": float(res.R_post),
                }
            )

    return rows


def _summarize_draws(draws: pd.DataFrame) -> pd.DataFrame:
    def q(p: float):
        return lambda s: float(np.nanquantile(s.to_numpy(), p))

    grp = draws.groupby(["compression_method", "lam"], as_index=False)

    out = grp.agg(
        n=("erls", "size"),
        erls_mean=("erls", "mean"),
        erls_p25=("erls", q(0.25)),
        erls_p50=("erls", q(0.50)),
        erls_p75=("erls", q(0.75)),
        kappa_mean=("kappa", "mean"),
        kappa_p25=("kappa", q(0.25)),
        kappa_p50=("kappa", q(0.50)),
        kappa_p75=("kappa", q(0.75)),
    )
    return out.sort_values(["compression_method", "lam"]).reset_index(drop=True)


@dataclass(frozen=True)
class Exp2RunResult:
    run_id: str
    artifact_dir: Path
    draws_csv: Path
    summary_csv: Path
    meta_json: Path
    spec_json: Path


def run_exp2(config: Exp2Config, draw_ids: Sequence[int], tag: str = "paper") -> Exp2RunResult:
    run_id = make_run_id("exp2", tag)
    artifact_dir = ARTIFACT_ROOT / "exp2" / run_id
    ensure_dir(artifact_dir)

    all_rows: List[Dict[str, Any]] = []
    for m in config.methods:
        solver = config.maxc_solver if m == "maxc" else None
        rows = run_exp2_method(
            config.spec,
            draw_ids=draw_ids,
            compression_method=m,
            compression_solver=solver,
        )
        all_rows.extend(rows)

    draws_df = pd.DataFrame(all_rows)
    summary_df = _summarize_draws(draws_df)

    draws_csv = artifact_dir / "draws.csv"
    summary_csv = artifact_dir / "summary.csv"
    meta_json = artifact_dir / "meta.json"
    spec_json = artifact_dir / "spec.json"

    rows_to_csv(all_rows, draws_csv)
    summary_df.to_csv(summary_csv, index=False)

    save_json(
        spec_json,
        {
            "config": {
                "methods": list(config.methods),
                "maxc_solver": config.maxc_solver,
                "use_lcc": bool(config.spec.network.use_lcc),

            },
            "experiment_spec": spec_to_jsonable(config.spec),
        },
    )

    save_json(
        meta_json,
        {
            "run_id": run_id,
            "experiment": "exp2",
            "tag": tag,
            "timestamp_utc": utc_now_iso(),
            "git": git_info(),
            "python": python_info(),
            "master_seed": int(PAPER_MASTER_SEED),
            "draw_ids": list(map(int, draw_ids)),
            "artifact_dir": str(artifact_dir),
        },
    )

    return Exp2RunResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        draws_csv=draws_csv,
        summary_csv=summary_csv,
        meta_json=meta_json,
        spec_json=spec_json,
    )

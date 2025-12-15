# experiments/runners/shortfall_vs_lambda_core.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv

import numpy as np

from src.experiment_specs import ExperimentSpec
from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import compress
from src.shocks import generate_uniform_factor_shapes, UniformShockShapeParams
from src.erls import xi_from_uniform_base
from src.buffers import behavioural_base_from_V
from src.simulation import run_fpa


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rows_to_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    _ensure_dir(out_csv)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_shortfall_vs_lambda_optionA(
    spec: ExperimentSpec,
    *,
    n_draws: int,
    # fixed-buffer design
    buffer_theta: float = 1.0,  # b = theta * behavioural_base_from_V(V)
    # compression override
    compression_method: str | None = None,
    compression_solver: str | None = None,
    # IO / seeds
    out_csv: str | Path | None = None,
    seed_offset_draws: int = 0,
    seed_offset_network: int | None = None,
    seed_offset_shock: int | None = None,
    use_lcc: bool = True,
    # plotting (optional)
    plot: bool = False,
    plot_title: str | None = None,
    out_png: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Experiment 1 core runner (fixed buffers):
      - draw network V and base shock shape U_base once per draw
      - fix buffers b_fixed = buffer_theta * behavioural_base_from_V(V)
      - for each lambda: build xi(lambda) and run FPA pre and post compression
      - record R_pre, R_post, and delta_R = R_pre - R_post (should be >= 0 typically)

    This quantifies "how much better off might you be if buffers don't change".
    """
    rows: List[Dict[str, Any]] = []

    method = compression_method or spec.compression.method
    solver = compression_solver or spec.compression.solver

    net_seed_base = seed_offset_network if seed_offset_network is not None else spec.network.seed_offset
    shock_seed_base = seed_offset_shock if seed_offset_shock is not None else spec.shock.seed_offset

    for k in range(n_draws):
        draw_id = seed_offset_draws + k

        # --- 1) draw network ---
        rng_net = np.random.default_rng(net_seed_base + draw_id)
        G = generate_three_tier_network(
            n_core=spec.network.n_core,
            n_source=spec.network.n_source,
            n_sink=spec.network.n_sink,
            p=spec.network.p,
            weight_mode=spec.network.weight_mode,
            alpha_weights=spec.network.alpha_weights,
            scale_weights=spec.network.scale_weights,
            rng=rng_net,
            degree_mode=spec.network.degree_mode,
            round_to=spec.network.round_to,
        )
        if use_lcc:
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

        # --- 3) fixed buffers (no recalibration) ---
        b_base = behavioural_base_from_V(V)  # typically row-sums, shape (N,) or (N,1)
        b_base = np.asarray(b_base, dtype=float).reshape(-1, 1)
        b_fixed = float(buffer_theta) * b_base

        # --- 4) base shock shape (fixed across lambda) ---
        U_base = generate_uniform_factor_shapes(
            num_nodes=N,
            params=UniformShockShapeParams(rho=spec.shock.rho_xi),
            n_samples=1,
            seed=shock_seed_base + draw_id,
        )[0].reshape(-1, 1)

        # --- 5) sweep lambda ---
        for lam in spec.shock.lam_grid:
            xi = xi_from_uniform_base(V_ref=V, U=U_base, lam=float(lam), scale=spec.shock.xi_scale)

            # initial liquidity for FPA
            b0 = b_fixed - xi

            fpa_pre = run_fpa(V, b0, rel_tol=spec.search.rel_tol_fpa, return_paths=False)
            fpa_post = run_fpa(V_tilde, b0, rel_tol=spec.search.rel_tol_fpa, return_paths=False)

            R_pre = float(fpa_pre.aggregate_shortfall)
            R_post = float(fpa_post.aggregate_shortfall)
            delta_R = R_pre - R_post

            # optional: normalize by total obligations to make plots scale-free
            gross_before = float(comp.gross_before)
            delta_R_norm = delta_R / gross_before if gross_before > 0 else float("nan")

            rows.append(
                {
                    "spec": spec.name,
                    "draw": int(draw_id),
                    "N": int(N),
                    "use_lcc": bool(use_lcc),
                    "compression_method": str(method),
                    "compression_solver": str(solver),
                    "buffer_theta": float(buffer_theta),
                    "rho_xi": float(spec.shock.rho_xi),
                    "xi_scale": spec.shock.xi_scale,
                    "lam": float(lam),
                    "R_pre": R_pre,
                    "R_post": R_post,
                    "delta_R": float(delta_R),
                    "delta_R_norm": float(delta_R_norm),
                    "gross_before": float(comp.gross_before),
                    "gross_after": float(comp.gross_after),
                    "savings_frac": float(comp.savings_frac),
                }
            )

    if out_csv is not None:
        out_csv = Path(out_csv)
        _rows_to_csv(rows, out_csv)

    if plot:
        from src.plots import plot_overlay_vs_lambda

        # By default plot delta_R (or delta_R_norm if you prefer)
        plot_overlay_vs_lambda(
            rows,
            group_key="compression_method",
            y_key="delta_R",
            y_label=r"$\Delta R(\lambda)=R_{pre}-R_{post}$",
            hline=0.0,
            title=plot_title or f"{spec.name} | Fixed buffers (θ={buffer_theta}) | ΔR vs λ",
            out_png=out_png,
        )

    return rows


def _example_main() -> None:
    from src.base_cases import BASE_CASE

    base = BASE_CASE
    n = 30
    theta = 0.8  # choose <1 if you want to ensure some positive shortfalls appear

    rows_bff = run_shortfall_vs_lambda_optionA(
        base,
        n_draws=n,
        buffer_theta=theta,
        compression_method="bff",
        out_csv=Path("figures") / f"{base.name}_exp1_theta{theta}_bff_n{n}.csv",
        plot=False,
    )

    rows_maxc = run_shortfall_vs_lambda_optionA(
        base,
        n_draws=n,
        buffer_theta=theta,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=Path("figures") / f"{base.name}_exp1_theta{theta}_maxc_n{n}.csv",
        plot=False,
    )

    from src.plots import plot_overlay_vs_lambda

    plot_overlay_vs_lambda(
        rows_bff + rows_maxc,
        group_key="compression_method",
        y_key="delta_R",
        y_label=r"$\Delta R(\lambda)$",
        hline=0.0,
        title=f"{base.name} | Exp1 fixed buffers (θ={theta}) | ΔR overlay",
        out_png=Path("figures") / f"{base.name}_exp1_theta{theta}_overlay_deltaR_n{n}.png",
    )


if __name__ == "__main__":
    _example_main()

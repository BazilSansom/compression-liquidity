# experiments/runners/erls_vs_lambda_core.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import csv

import numpy as np

from src.experiment_specs import ExperimentSpec
from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import compress
from src.shocks import generate_uniform_factor_shapes, UniformShockShapeParams
from src.erls import compute_erls_zero_shortfall



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


def run_erls_vs_lambda_optionA(
    spec: ExperimentSpec,
    *,
    n_draws: int,
    compression_method: str | None = None,
    compression_solver: str | None = None,
    out_csv: str | Path | None = None,
    seed_offset_draws: int = 0,
    seed_offset_network: int | None = None,
    seed_offset_shock: int | None = None,
    use_lcc: bool = True,
    plot: bool = False,
    plot_kappa: bool = False,   
    plot_title: str | None = None,
    out_png: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Core runner for Experiment 3-style ERLS vs lambda (Option A sampling).

    Option A sampling:
      - for each draw k: sample one network G and one base shock shape U_base
      - for each lambda in spec.shock.lam_grid: compute ERLS holding (G, U_base) fixed

    Inputs:
      - compression_method: override spec.compression.method locally (e.g. "bff", "maxc")
      - compression_solver: optional override for solver (e.g. "ortools" for maxc)
      - seed_offset_draws: shifts draw ids (useful for chunking)
      - seed_offset_network / seed_offset_shock: optional overrides to make runs reproducible
      - plot: optionally call src.plots.quick_plot_erls_vs_lambda at end (single-series sanity plot)

    Returns:
      - rows: list of dicts (also optionally written to CSV)
    """
    rows: List[Dict[str, Any]] = []

    # Local overrides (do not mutate spec)
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
        N = int(G.num_nodes)  # IMPORTANT: may differ from spec.network.N if use_lcc=True

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
            seed=shock_seed_base + draw_id,
        )[0].reshape(-1, 1)


        # --- 4) sweep lambda ---
        for lam in spec.shock.lam_grid:
            res = compute_erls_zero_shortfall(
                V=V,
                V_tilde=V_tilde,
                U_base=U_base,
                lam=float(lam),
                xi_scale=spec.shock.xi_scale,
                buffer_mode=spec.buffer_mode,
                rel_tol_fpa=spec.search.rel_tol_fpa,
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
                    "draw": int(draw_id),
                    "N": int(N),
                    "use_lcc": bool(use_lcc),
                    "compression_method": str(method),
                    "compression_solver": str(solver),
                    "rho_xi": float(spec.shock.rho_xi),
                    "xi_scale": spec.shock.xi_scale,
                    "buffer_mode": spec.buffer_mode,
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

    # --- write CSV (optional) ---
    if out_csv is not None:
        out_csv = Path(out_csv)
        _rows_to_csv(rows, out_csv)

    # --- quick plots (optional) ---
    if plot:
        from src.plots import quick_plot_vs_lambda

        # ERLS
        quick_plot_vs_lambda(
            rows,
            y_key="erls",
            y_label="ERLS",
            hline=0.0,
            title=plot_title or f"{spec.name} | {method} | {spec.buffer_mode}",
            out_png=out_png,
        )

        # kappa
        if plot_kappa:
            quick_plot_vs_lambda(
                rows,
                y_key="kappa",
                y_label=r"$\kappa$ (required scale factor)",
                hline=1.0,
                title=(plot_title or f"{spec.name} | {method} | {spec.buffer_mode}") + " — κ",
                out_png=None,
            )

    return rows


# ---------------------------
# Example usage (optional)
# ---------------------------
def _example_main() -> None:
    from src.base_cases import BASE_CASE

    spec = BASE_CASE
    n = 50

    # Run BFF
    rows_bff = run_erls_vs_lambda_optionA(
        spec,
        n_draws=n,
        compression_method="bff",
        out_csv=Path("figures") / f"{spec.name}_bff_n{n}.csv",
        plot=True,
        plot_kappa=True,
        out_png=Path("figures") / f"{spec.name}_bff_n{n}.png",
    )

    # Run maxC (ortools)
    rows_maxc = run_erls_vs_lambda_optionA(
        spec,
        n_draws=n,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=Path("figures") / f"{spec.name}_maxc_n{n}.csv",
        plot=True,
        plot_kappa=True,
        out_png=Path("figures") / f"{spec.name}_maxc_n{n}.png",
    )

    rows_all = rows_bff + rows_maxc

    # Overlay plots (concatenate and group)
    from src.plots import plot_overlay_erls_vs_lambda, plot_overlay_vs_lambda

    # ERLS overlay (existing)
    plot_overlay_erls_vs_lambda(
        rows_all,
        group_key="compression_method",
        title=f"{spec.name} | ERLS overlay",
        out_png=Path("figures") / f"{spec.name}_overlay_erls_n{n}.png",
    )

    # kappa overlay (new)
    plot_overlay_vs_lambda(
        rows_all,
        group_key="compression_method",
        y_key="kappa",
        y_label=r"$\kappa$ (required scale factor)",
        hline=1.0,
        title=f"{spec.name} | κ overlay",
        out_png=Path("figures") / f"{spec.name}_overlay_kappa_n{n}.png",
    )


if __name__ == "__main__":
    _example_main()



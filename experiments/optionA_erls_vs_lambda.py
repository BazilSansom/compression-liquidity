# experiments/optionA_erls_vs_lambda.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.base_cases import BASE_CASE
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


def run_optionA_erls_vs_lambda(
    spec: ExperimentSpec,
    *,
    n_draws: int,
    out_csv: str | Path = "figures/optionA_erls_vs_lambda.csv",
    seed_offset_draws: int = 0,
    use_lcc: bool = True,
) -> List[Dict[str, Any]]:
    """
    Option A sampling:
      - for each draw k: sample one network G and one base shock U_base
      - for each lambda in spec.shock.lam_grid: compute ERLS holding (G, U_base) fixed

    If use_lcc=True, we restrict the network to its largest weakly connected component
    to avoid degenerate isolated nodes/components.
    """
    out_csv = Path(out_csv)
    rows: List[Dict[str, Any]] = []

    for k in range(n_draws):
        draw_id = seed_offset_draws + k

        # --- 1) draw network ---
        rng_net = np.random.default_rng(spec.network.seed_offset + draw_id)
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

        if spec.network.use_lcc:
            G = extract_largest_component(G)

        V = np.asarray(G.W, dtype=float)
        N = G.num_nodes  # IMPORTANT: may differ from spec.network.N if use_lcc=True

        # --- 2) compress ---
        comp = compress(
            G,
            method=spec.compression.method,
            solver=spec.compression.solver,
            tol_zero=spec.compression.tol_zero,
            require_conservative=spec.compression.require_conservative,
            require_full_conservative=spec.compression.require_full_conservative,
        )
        V_tilde = np.asarray(comp.compressed.W, dtype=float)

        # --- 3) draw base shock U_base once (fixed across lambda) ---
        U_base = generate_uniform_factor_shapes(
            num_nodes=N,
            params=UniformShockShapeParams(rho=spec.shock.rho_xi),
            n_samples=1,
            seed=spec.shock.seed_offset + draw_id,
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

            rows.append(
                {
                    "spec": spec.name,
                    "draw": int(draw_id),
                    "N": int(N),
                    "use_lcc": bool(use_lcc),
                    "compression_method": comp.method,  # e.g. "BFF" or "maxC_ortools"
                    "rho_xi": float(spec.shock.rho_xi),
                    "xi_scale": spec.shock.xi_scale,
                    "buffer_mode": spec.buffer_mode,
                    "lam": float(lam),
                    "alpha_pre": float(res.alpha_pre),
                    "alpha_post": float(res.alpha_post),
                    "erls": float(res.erls),
                    "R_pre": float(res.R_pre),
                    "R_post": float(res.R_post),
                    "gross_before": float(comp.gross_before),
                    "gross_after": float(comp.gross_after),
                    "savings_frac": float(comp.savings_frac),
                }
            )

    _rows_to_csv(rows, out_csv)
    return rows


def quick_plot_erls_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    title: str | None = None,
    out_png: str | Path | None = None,
) -> None:
    """
    Quick sanity plot: mean ERLS vs lambda with IQR band.
    """
    grouped = defaultdict(list)
    for r in rows:
        grouped[float(r["lam"])].append(float(r["erls"]))

    lams = sorted(grouped.keys())
    means, q25, q75 = [], [], []

    for lam in lams:
        vals = np.array(grouped[lam], dtype=float)
        means.append(vals.mean())
        q25.append(np.percentile(vals, 25))
        q75.append(np.percentile(vals, 75))

    plt.figure()
    plt.plot(lams, means, marker="o", label="Mean ERLS")
    plt.fill_between(lams, q25, q75, alpha=0.3, label="IQR")
    plt.axhline(0.0, linewidth=1, linestyle="--")
    plt.xlabel(r"$\lambda$ (liquidity shock intensity)")
    plt.ylabel("ERLS")
    plt.title(title or "ERLS vs liquidity shock intensity")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)

    plt.show()


def main() -> None:
    spec = BASE_CASE
    n_draws = 50

    out_csv = Path("figures") / f"{spec.name}_n{n_draws}_optionA_erls_vs_lambda.csv"
    out_png = Path("figures") / f"{spec.name}_n{n_draws}_erls_vs_lambda.png"

    rows = run_optionA_erls_vs_lambda(
        spec,
        n_draws=n_draws,
        out_csv=out_csv,
        use_lcc=True,
    )

    quick_plot_erls_vs_lambda(
        rows,
        title=f"{spec.name} | {spec.compression.method.upper()} | {spec.buffer_mode}",
        out_png=out_png,
    )


if __name__ == "__main__":
    main()

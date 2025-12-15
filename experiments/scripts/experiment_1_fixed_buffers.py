# experiments/scripts/experiment_1_fixed_buffers.py
from __future__ import annotations

from dataclasses import replace

from experiments.runners.shortfall_vs_lambda_core import run_shortfall_vs_lambda_optionA
from src.plots import plot_overlay_vs_lambda
from experiments.scripts.paper_config import (
    FIG_ROOT,
    N_DRAWS,
    PAPER_BASE_SPEC,
    THETA_FIXED_BUFFERS,
)


def main() -> None:
    base = PAPER_BASE_SPEC

    # Fixed-buffer benchmark (Experiment 1)
    theta = THETA_FIXED_BUFFERS
    n = N_DRAWS

    theta_tag = f"{theta:g}"
    spec = replace(base, name=f"{base.name}_exp1_fixed_theta{theta_tag}")

    fig_dir = FIG_ROOT / "exp1"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows_bff = run_shortfall_vs_lambda_optionA(
        spec,
        n_draws=n,
        buffer_theta=theta,
        compression_method="bff",
        out_csv=fig_dir / f"{spec.name}_bff_n{n}.csv",
        plot=False,
    )

    rows_maxc = run_shortfall_vs_lambda_optionA(
        spec,
        n_draws=n,
        buffer_theta=theta,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=fig_dir / f"{spec.name}_maxc_n{n}.csv",
        plot=False,
    )

    rows_all = rows_bff + rows_maxc

    # === Normalised realised shortfall reduction (main paper object) ===
    plot_overlay_vs_lambda(
        rows_all,
        group_key="compression_method",
        y_key="delta_R_norm",
        y_label=r"$(R_{\mathrm{pre}}-R_{\mathrm{post}})/\sum_{ij} V_{ij}$",
        hline=0.0,
        title=r"Normalized shortfall reduction vs $\lambda$",
        out_png=fig_dir / "exp1_normalized_shortfall_reduction_vs_lambda.png",
    )


if __name__ == "__main__":
    main()

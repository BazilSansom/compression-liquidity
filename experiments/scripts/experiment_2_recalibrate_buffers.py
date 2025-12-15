# experiments/scripts/experiment_2_recalibrate_buffers.py
from __future__ import annotations

from dataclasses import replace

from experiments.runners.erls_vs_lambda_core import run_erls_vs_lambda_optionA
from src.plots import plot_overlay_vs_lambda
from experiments.scripts.paper_config import FIG_ROOT, N_DRAWS, PAPER_BASE_SPEC


def main() -> None:
    base = PAPER_BASE_SPEC
    n = N_DRAWS

    # Behavioural buffer recalibration (Experiment 2 in the paper)
    spec = replace(base, name=f"{base.name}_exp2_behavioural", buffer_mode="behavioural")

    fig_dir = FIG_ROOT / "exp2"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows_bff = run_erls_vs_lambda_optionA(
        spec,
        n_draws=n,
        compression_method="bff",
        out_csv=fig_dir / f"{spec.name}_bff_n{n}.csv",
        plot=False,
    )

    rows_maxc = run_erls_vs_lambda_optionA(
        spec,
        n_draws=n,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=fig_dir / f"{spec.name}_maxc_n{n}.csv",
        plot=False,
    )

    rows_all = rows_bff + rows_maxc

    # === ERLS (main paper figure) ===
    plot_overlay_vs_lambda(
        rows_all,
        group_key="compression_method",
        y_key="erls",
        y_label="ERLS",
        hline=0.0,
        title=r"Equal-risk liquidity savings vs $\lambda$",
        out_png=fig_dir / "exp2_erls_vs_lambda.png",
    )

    # === κ (important behavioural diagnostic) ===
    plot_overlay_vs_lambda(
        rows_all,
        group_key="compression_method",
        y_key="kappa",
        y_label=r"$\kappa$ (required buffer scale factor)",
        hline=1.0,
        title=r"Conservativeness ratio $\kappa$ vs $\lambda$",
        out_png=fig_dir / "exp2_kappa_vs_lambda.png",
    )


if __name__ == "__main__":
    main()


#experiments/scripts/experiment_3_behavioural.py

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from src.base_cases import BASE_CASE
from experiments.runners.erls_vs_lambda_core import run_erls_vs_lambda_optionA
from src.plots import plot_overlay_erls_vs_lambda, plot_overlay_vs_lambda


def main() -> None:
    base = BASE_CASE

    # Ensure behavioural mode (Experiment 3)
    spec = replace(base, name=f"{base.name}_exp3_behavioural", buffer_mode="behavioural")

    n = 20  # bump later
    fig_dir = Path("figures") / "exp3"
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

    # ERLS overlay
    plot_overlay_erls_vs_lambda(
        rows_all,
        group_key="compression_method",
        title=f"{spec.name} | ERLS vs λ (behavioural)",
        out_png=fig_dir / f"{spec.name}_overlay_erls_n{n}.png",
    )

    # kappa overlay (important for behavioural story)
    plot_overlay_vs_lambda(
        rows_all,
        group_key="compression_method",
        y_key="kappa",
        y_label=r"$\kappa$ (required scale factor)",
        hline=1.0,
        title=f"{spec.name} | κ vs λ (behavioural)",
        out_png=fig_dir / f"{spec.name}_overlay_kappa_n{n}.png",
    )


if __name__ == "__main__":
    main()

# experiments/optionA_erls_vs_lambda_overlay.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

import time

from src.base_cases import BASE_CASE
from src.experiment_specs import ExperimentSpec
from experiments.optionA_erls_vs_lambda import run_optionA_erls_vs_lambda


def _group_stats(rows: List[Dict[str, Any]], key: str) -> dict[str, dict[float, dict[str, float]]]:
    """
    Return stats[group][lam] = {mean, q25, q75, n}
    """
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[str(r[key])][float(r["lam"])].append(float(r["erls"]))

    stats: dict[str, dict[float, dict[str, float]]] = {}
    for g, by_lam in grouped.items():
        stats[g] = {}
        for lam, vals in by_lam.items():
            arr = np.asarray(vals, dtype=float)
            stats[g][lam] = {
                "mean": float(arr.mean()),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "n": int(arr.size),
            }
    return stats


def plot_overlay_erls_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    group_key: str = "compression_method",
    title: str | None = None,
    out_png: str | Path | None = None,
) -> None:
    """
    Overlay mean ERLS vs lambda for multiple groups (e.g. compression_method),
    with an IQR band per group.
    """
    stats = _group_stats(rows, key=group_key)
    groups = sorted(stats.keys())

    plt.figure()

    for g in groups:
        lams = sorted(stats[g].keys())
        mean = [stats[g][lam]["mean"] for lam in lams]
        q25 = [stats[g][lam]["q25"] for lam in lams]
        q75 = [stats[g][lam]["q75"] for lam in lams]

        plt.plot(lams, mean, marker="o", label=f"{g} mean")
        plt.fill_between(lams, q25, q75, alpha=0.2, label=f"{g} IQR")

    plt.axhline(0.0, linewidth=1, linestyle="--")
    plt.xlabel(r"$\lambda$ (liquidity shock intensity)")
    plt.ylabel("ERLS")
    plt.title(title or f"ERLS vs lambda (grouped by {group_key})")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)

    plt.show()



def main() -> None:
    
    t0 = time.perf_counter()

    # Base spec (your current setup)
    base: ExperimentSpec = BASE_CASE

    # Two variants: BFF vs maxC
    spec_bff = replace(base, name=f"{base.name}_bff", compression=replace(base.compression, method="bff"))
    spec_maxc = replace(
        base,
        name=f"{base.name}_maxc",
        compression=replace(base.compression, method="maxc", solver="ortools"),
    )

    n_draws = 50  # start with 50–200
    out_dir = Path("figures")

    rows_bff = run_optionA_erls_vs_lambda(
        spec_bff,
        n_draws=n_draws,
        out_csv=out_dir / f"{spec_bff.name}_optionA_erls_vs_lambda.csv",
    )
    rows_maxc = run_optionA_erls_vs_lambda(
        spec_maxc,
        n_draws=n_draws,
        out_csv=out_dir / f"{spec_maxc.name}_optionA_erls_vs_lambda.csv",
    )

    rows_all = rows_bff + rows_maxc

    plot_overlay_erls_vs_lambda(
        rows_all,
        group_key="compression_method",
        title=f"{base.name} | behavioural buffers | BFF vs maxC (n_draws={n_draws})",
        out_png=out_dir / f"{base.name}_overlay_bff_vs_maxc.png",
    )

    t1 = time.perf_counter()

    print("\n=== Runtime summary ===")
    print(f"Draws            : {n_draws}")
    print(f"Lambdas per draw : {len(base.shock.lam_grid)}")
    print(f"Total scenarios  : {n_draws * len(base.shock.lam_grid) * 2}  (BFF + maxC)")
    print(f"Wall time        : {t1 - t0:.2f} seconds")
    print("=======================\n")


if __name__ == "__main__":
    main()

# src/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def quick_plot_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    y_key: str = "erls",
    y_label: str | None = None,
    title: str | None = None,
    out_png: str | Path | None = None,
    hline: float | None = 0.0,
) -> None:
    """
    Quick sanity plot: mean(y_key) vs lambda with IQR band.
    """
    if not rows:
        raise ValueError("No rows to plot.")
    if y_key not in rows[0]:
        raise KeyError(f"y_key='{y_key}' not found. Available keys: {sorted(rows[0].keys())}")

    grouped = defaultdict(list)
    for r in rows:
        grouped[float(r["lam"])].append(float(r[y_key]))

    lams = sorted(grouped.keys())
    means, q25, q75 = [], [], []

    for lam in lams:
        vals = np.asarray(grouped[lam], dtype=float)
        means.append(vals.mean())
        q25.append(np.percentile(vals, 25))
        q75.append(np.percentile(vals, 75))

    plt.figure()
    plt.plot(lams, means, marker="o", label=f"Mean {y_key}")
    plt.fill_between(lams, q25, q75, alpha=0.3, label="IQR")
    if hline is not None:
        plt.axhline(hline, linewidth=1, linestyle="--")
    plt.xlabel(r"$\lambda$ (liquidity shock intensity)")
    plt.ylabel(y_label or y_key)
    plt.title(title or f"{y_key} vs liquidity shock intensity")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        _ensure_dir(out_png)
        plt.savefig(out_png, dpi=200)

    plt.show()


def _group_stats(
    rows: List[Dict[str, Any]],
    *,
    group_key: str,
    y_key: str,
) -> dict[str, dict[float, dict[str, float]]]:
    """
    Return stats[group][lam] = {mean, q25, q75, n} for the metric y_key.
    """
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[str(r[group_key])][float(r["lam"])].append(float(r[y_key]))

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


def plot_overlay_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    group_key: str = "compression_method",
    y_key: str = "erls",
    y_label: str | None = None,
    title: str | None = None,
    out_png: str | Path | None = None,
    show_n: bool = False,
    hline: float | None = 0.0,
) -> None:
    """
    Overlay mean(y_key) vs lambda for multiple groups (e.g. compression_method),
    with an IQR band per group.
    """
    if not rows:
        raise ValueError("No rows to plot.")
    if y_key not in rows[0]:
        raise KeyError(f"y_key='{y_key}' not found. Available keys: {sorted(rows[0].keys())}")


    stats = _group_stats(rows, group_key=group_key, y_key=y_key)
    groups = sorted(stats.keys())

    plt.figure()

    for g in groups:
        lams = sorted(stats[g].keys())
        mean = [stats[g][lam]["mean"] for lam in lams]
        q25 = [stats[g][lam]["q25"] for lam in lams]
        q75 = [stats[g][lam]["q75"] for lam in lams]

        label = f"{g}"
        if show_n:
            # show n at the first lambda (assumes constant n across lambda in Option A)
            n0 = stats[g][lams[0]]["n"]
            label = f"{g} (n={int(n0)})"

        plt.plot(lams, mean, marker="o", label=label)
        plt.fill_between(lams, q25, q75, alpha=0.2)

    if hline is not None:
        plt.axhline(hline, linewidth=1, linestyle="--")
    plt.xlabel(r"$\lambda$ (liquidity shock intensity)")
    plt.ylabel(y_label or y_key)
    plt.title(title or f"{y_key} vs lambda (grouped by {group_key})")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        _ensure_dir(out_png)
        plt.savefig(out_png, dpi=200)

    plt.show()


# Backwards-compatible aliases (optional)
def quick_plot_erls_vs_lambda(rows: List[Dict[str, Any]], *, title: str | None = None, out_png: str | Path | None = None) -> None:
    quick_plot_vs_lambda(rows, y_key="erls", y_label="ERLS", title=title, out_png=out_png)


def plot_overlay_erls_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    group_key: str = "compression_method",
    title: str | None = None,
    out_png: str | Path | None = None,
) -> None:
    plot_overlay_vs_lambda(rows, group_key=group_key, y_key="erls", y_label="ERLS", title=title, out_png=out_png)

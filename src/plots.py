# src/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.stats import lorenz_points, topk_shares


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



def _group_stats_multi(
    rows: List[Dict[str, Any]],
    *,
    group_key: str,
    y_keys: Sequence[str],
    x_key: str = "lam",
) -> Dict[str, Dict[float, Dict[str, Dict[str, float]]]]:
    """
    Returns nested dict:
    stats[group][lam][y_key] = {"mean":..., "q25":..., "q75":..., "n":...}
    """
    out: Dict[str, Dict[float, Dict[str, Dict[str, float]]]] = {}

    for r in rows:
        g = str(r[group_key])
        lam = float(r[x_key])

        out.setdefault(g, {})
        out[g].setdefault(lam, {})
        for yk in y_keys:
            out[g][lam].setdefault(yk, {"_vals": []})
            out[g][lam][yk]["_vals"].append(float(r[yk]))

    # reduce
    for g in out:
        for lam in out[g]:
            for yk in y_keys:
                vals = np.asarray(out[g][lam][yk].pop("_vals"), dtype=float)
                out[g][lam][yk] = {
                    "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                    "q25": float(np.percentile(vals, 25)) if len(vals) else float("nan"),
                    "q75": float(np.percentile(vals, 75)) if len(vals) else float("nan"),
                    "n": float(len(vals)),
                }

    return out


def plot_stacked_shares_vs_lambda(
    rows: List[Dict[str, Any]],
    *,
    group_key: str = "compression_method",
    
    # defaults
    share_keys: Sequence[str] = (
        "share_new_trigger",
        "share_new_late",
        "share_still_inactive_cash_relief",
        "share_still_inactive_obligation_relief",
    ),

    share_labels: Sequence[str] | None = None,
    title: str | None = None,
    out_png: str | Path | None = None,
    # layout controls
    method: str | None = None,
    two_panel: bool = True,
    # diagnostics / cosmetics
    show_n: bool = True,
    eps: float = 1e-12,          # draws with delta_R <= eps are skipped
    renormalize: bool = True,    # renormalize tiny floating drift so stack hits 1
) -> None:
    """
    Stacked shares vs lambda, computed as ratio-of-means:
        share_m(lam) = sum_draw ΔR_m / sum_draw ΔR
    (skipping draws with ΔR ~ 0).
    This guarantees the stack totals ~1.
    """
    if not rows:
        raise ValueError("No rows to plot.")

    # map share keys -> delta keys
    share_to_delta = {
        # New/active decomposition
        "share_new_trigger": "delta_R_new_trigger",
        "share_new_late": "delta_R_new_late",
        "share_new_active": "delta_R_new_active",

        # OLD inactive split (kept for backwards compatibility)
        "share_still_inactive_new_payments_inflow": "delta_R_still_inactive_new_payments_inflow",
        "share_still_inactive_direct_netting_relief": "delta_R_still_inactive_direct_netting_relief",

        # NEW inactive split (cash vs obligation relief)
        "share_still_inactive_cash_relief": "delta_R_still_inactive_cash_relief",
        "share_still_inactive_obligation_relief": "delta_R_still_inactive_obligation_relief",
    }

    delta_keys = []
    for sk in share_keys:
        if sk in share_to_delta:
            delta_keys.append(share_to_delta[sk])
        else:
            raise KeyError(
                f"Unrecognized share key '{sk}'. "
                f"Known: {sorted(share_to_delta.keys())}. "
                f"(If you add a new share, extend share_to_delta.)"
            )

    denom_key = "delta_R_raw"  # total improvement

    # labels
    if share_labels is None:
        share_labels = list(share_keys)

    # filter to one method if requested
    plot_rows = rows
    if method is not None:
        plot_rows = [r for r in rows if str(r[group_key]) == str(method)]
        if not plot_rows:
            raise ValueError(f"No rows found for {group_key}='{method}'.")

    # --- aggregate sums by (group, lam) ---
    agg: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for r in plot_rows:
        g = str(r[group_key])
        lam = float(r["lam"])

        dR = float(r.get(denom_key, 0.0))
        if not np.isfinite(dR) or dR <= eps:
            continue  # skip ill-defined share draws

        agg.setdefault(g, {}).setdefault(lam, {"sum_dR": 0.0, "sum_parts": [0.0]*len(delta_keys), "n": 0})
        agg[g][lam]["sum_dR"] += dR

        for i, dk in enumerate(delta_keys):
            val = float(r.get(dk, 0.0))
            if np.isfinite(val):
                agg[g][lam]["sum_parts"][i] += val
        agg[g][lam]["n"] += 1

    if not agg:
        raise ValueError("After filtering ΔR≈0 draws, no data left to plot. Try lowering eps.")

    groups = sorted(agg.keys())

    # decide layout
    if method is not None or not two_panel:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        axes = [ax]
        panel_groups = [groups] if method is None else [[groups[0]]]
    else:
        n_panels = len(groups)
        fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5), sharey=True)
        if n_panels == 1:
            axes = [axes]
        panel_groups = [[g] for g in groups]

    for ax, g_list in zip(axes, panel_groups):
        for g in g_list:
            lams = sorted(agg[g].keys())
            shares_by_comp = [[] for _ in delta_keys]
            ns = []

            for lam in lams:
                tot = agg[g][lam]["sum_dR"]
                parts = agg[g][lam]["sum_parts"]
                n = agg[g][lam]["n"]
                ns.append(n)

                if tot <= eps:
                    sh = [np.nan]*len(parts)
                else:
                    sh = [p / tot for p in parts]
                    if renormalize:
                        s = sum(sh)
                        if np.isfinite(s) and s > 0:
                            sh = [x / s for x in sh]

                for i in range(len(parts)):
                    shares_by_comp[i].append(sh[i])

            y_stack = [np.asarray(v, dtype=float) for v in shares_by_comp]

            ax.stackplot(lams, y_stack, labels=share_labels, alpha=0.85)

            if show_n and len(lams) > 0:
                ax.set_title(f"{g} (n={int(ns[0])})")
            else:
                ax.set_title(str(g))

        ax.set_xlabel(r"$\lambda$ (liquidity shock intensity)")
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("mechanism share")

    # single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(share_keys), frameon=True)

    fig.suptitle(title or "Mechanism shares vs $\\lambda$", y=1.05)
    fig.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        _ensure_dir(out_png)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")

    plt.show()




# ------- Activation pre/post plots ---------------

def plot_activation_pre_post(rows, out_png):
    # rows: list of dicts

    # group by lambda, then by method
    lams = sorted({r["lam"] for r in rows})

    # pre: average across all rows at each lambda (should be same across methods anyway)
    pre = []
    for lam in lams:
        vals = [r["share_active_pre"] for r in rows if r["lam"] == lam]
        pre.append(float(np.mean(vals)))

    plt.figure()
    plt.plot(lams, pre, marker="o", label="pre (uncompressed)")

    for method in sorted({r["compression_method"] for r in rows}):
        post = []
        for lam in lams:
            vals = [r["share_active_post"] for r in rows if r["lam"] == lam and r["compression_method"] == method]
            post.append(float(np.mean(vals)))
        plt.plot(lams, post, marker="o", label=f"post ({method})")

    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$\lambda$ (liquidity shock intensity)")
    plt.ylabel(r"Activation share $|A|/N$")
    plt.title(r"Activation share vs $\lambda$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------- Lorenz curvs ---------------

def plot_lorenz_from_long_csv(
    *,
    long_csv: str | Path,
    out_dir: str | Path,
    title: str = "Node-impact concentration (Lorenz curves)",
    compressions: tuple[str, ...] = ("unc", "bff", "maxc"),
    dpi: int = 200,
) -> Tuple[Path, Path]:
    """
    Read the long node-impact CSV and produce:
      1) Lorenz curve plot of per-node mean knockout impacts, by compression.
      2) A CSV of top-k impact shares (based on per-node mean impacts).

    Parameters
    ----------
    long_csv : path-like
        CSV produced by node impact runner (one row per xi_draw, compression, node).
        Must contain columns: compression, node, delta.
    out_dir : path-like
        Output directory for the PNG and top-k CSV.
    title : str
        Figure title.
    compressions : tuple[str,...]
        Order of compression labels to plot (if present).
    dpi : int
        DPI for saved PNG.

    Returns
    -------
    (out_png, out_topk_csv) : (Path, Path)
    """
    long_csv = Path(long_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(long_csv)

    required = {"compression", "node", "delta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {long_csv}")

    # Compute mean impact per node, per compression: \bar{Δ}_i
    mean_by_node = (
        df.groupby(["compression", "node"], as_index=False)["delta"]
        .mean()
        .rename(columns={"delta": "mean_delta"})
    )

    # Equality line
    x_eq = np.linspace(0.0, 1.0, 200)

    plt.figure()
    plt.plot(x_eq, x_eq, linestyle="--", label="equality")

    present = [c for c in compressions if c in set(mean_by_node["compression"])]
    if not present:
        raise ValueError(f"No expected compression labels {compressions} found in {long_csv}")

    topk_rows: list[dict[str, float]] = []

    for comp in present:
        vals = mean_by_node.loc[mean_by_node["compression"] == comp, "mean_delta"].to_numpy()
        x, y = lorenz_points(vals)
        plt.plot(x, y, label=comp)

        row: dict[str, float] = {"compression": comp}
        row.update(topk_shares(vals, ks=(1, 5, 10, 20)))
        topk_rows.append(row)

    plt.xlabel("Cumulative share of nodes")
    plt.ylabel("Cumulative share of total mean knockout impact")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_png = out_dir / f"lorenz_{long_csv.stem}.png"
    plt.savefig(out_png, dpi=dpi)
    plt.close()

    topk_df = pd.DataFrame(topk_rows)
    out_topk_csv = out_dir / f"topk_shares_{long_csv.stem}.csv"
    topk_df.to_csv(out_topk_csv, index=False)

    return out_png, out_topk_csv


# Optional CLI for ad-hoc use
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--long_csv",
        type=str,
        default="figures/paper/node_impact/phase1/long_n10_net0_shock0.csv",
        help="Path to the long CSV produced by the node impact runner.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="figures/paper/node_impact/phase1",
        help="Directory to write outputs (png + topk csv).",
    )
    ap.add_argument("--title", type=str, default="Node-impact concentration (Lorenz curves)")
    args = ap.parse_args()

    out_png, out_topk = plot_lorenz_from_long_csv(
        long_csv=args.long_csv,
        out_dir=args.out_dir,
        title=args.title,
    )
    print(f"Wrote:\n  {out_png}\n  {out_topk}")
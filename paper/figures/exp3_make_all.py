# paper/figures/exp3_make_all.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from paper.io import ensure_dir
from paper.paths import FIG_ROOT, TABLE_ROOT, PAPER_OUTPUT_ROOT


ZERO_TOL = 1e-12

@dataclass(frozen=True)
class Exp3FigurePaths:
    out_dir: Path
    scatter_core_png: Path
    scatter_core_pdf: Path
    scatter_core_demeaned_binned_png: Path
    scatter_core_demeaned_binned_pdf: Path
    hist_core_png: Path
    hist_core_pdf: Path
    reg_core_txt: Path
    reg_core_tex: Path
    reg_core_table_tex: Path

def _shares_three_way(x: np.ndarray, *, zero_tol: float) -> dict:
    x = x[np.isfinite(x)]
    return {
        "zero_tol": float(zero_tol),
        "n": int(x.size),
        "share_neg": float(np.mean(x < -zero_tol)),
        "share_zero": float(np.mean(np.abs(x) <= zero_tol)),
        "share_pos": float(np.mean(x > zero_tol)),
    }


def _term_stats(model, term: str) -> dict:
    """Return dict with coef/se/p, or nulls if absent."""
    if term not in model.params.index:
        return {"coef": None, "se": None, "p": None}
    return {
        "coef": float(model.params[term]),
        "se": float(model.bse[term]),
        "p": float(model.pvalues[term]),
    }


def _model_summary(model, *, name: str, formula: str, terms: list[str]) -> dict:
    return {
        "name": name,
        "formula": formula,
        "n_obs": int(model.nobs),
        "rsquared": float(model.rsquared),
        "terms": {t: _term_stats(model, t) for t in terms},
    }


#def _ensure_dir(p: Path) -> None:
#    p.mkdir(parents=True, exist_ok=True)


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _summary_to_tex(model, title: str) -> str:
    try:
        body = model.summary().as_latex()
    except Exception:
        body = str(model.summary())
    return "\n".join(
        [
            "% ================================================",
            f"% {title}",
            "% ================================================",
            body,
            "",
        ]
    )


def _cluster_ols(formula: str, df: pd.DataFrame, cluster_col: str):
    """OLS with cluster-robust SEs by cluster_col."""
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )


def _prepare_core(core: pd.DataFrame, *, eps: float = 1e-12) -> pd.DataFrame:
    """
    Add normalized slack, log controls, (optional) normalized HHI change,
    and within-network demeaned variants.
    """
    core = core.copy()

    # main regressor: normalized slack gap using BFF outgoing obligations
    core["delta_slack_frac"] = core["delta_slack_common"].astype(float) / (
        core["ell_bff"].astype(float) + eps
    )

    # optional concentration regressor: proportional change in outgoing HHI
    if "delta_hhi_out" in core.columns and "hhi_out_bff" in core.columns:
        core["delta_hhi_out_frac"] = core["delta_hhi_out"].astype(float) / (
            core["hhi_out_bff"].astype(float) + eps
        )

    # size-ish controls (log are numerically nicer)
    core["log_ell_bff"] = np.log1p(core["ell_bff"].astype(float))
    if "in_unc" in core.columns:
        core["log_in_unc"] = np.log1p(core["in_unc"].astype(float))

    # within-network demeaned versions (FE-equivalent slopes)
    dm_cols = [
        "delta_vuln_common",
        "delta_slack_frac",
        "delta_hhi_out_frac",
        "incoming_reliance_bff",
        "log_ell_bff",
        "log_in_unc",
    ]
    for c in dm_cols:
        if c in core.columns:
            core[c + "_dm"] = core[c] - core.groupby("net_id")[c].transform("mean")

    return core


def _scatter_with_fit(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    add_binned_means: bool = True,
    n_bins: int = 20,
    show_legend: bool = True,
) -> None:
    """
    Raw scatter + OLS fit line + optional binned means line (quantile bins on x).
    """
    fig = plt.figure()

    X = df[x].to_numpy()
    Y = df[y].to_numpy()
    mask = np.isfinite(X) & np.isfinite(Y)

    plt.scatter(X[mask], Y[mask], alpha=0.25, label="nodes")

    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(0.0, linewidth=1.0)

    # OLS fit line (simple polyfit)
    if mask.sum() >= 3:
        b, a = np.polyfit(X[mask], Y[mask], 1)
        xgrid = np.linspace(np.nanmin(X[mask]), np.nanmax(X[mask]), 200)
        yhat = a + b * xgrid
        plt.plot(xgrid, yhat, linewidth=2.0, label="OLS fit")

    # Binned means line (your “orange line”)
    if add_binned_means and mask.sum() >= max(20, n_bins):
        xvals = X[mask]
        yvals = Y[mask]

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(xvals, qs)
        edges = np.unique(edges)
        if edges.size >= 4:
            bin_id = np.digitize(xvals, edges[1:-1], right=True)
            xb, yb = [], []
            for bidx in range(edges.size - 1):
                sel = bin_id == bidx
                if np.any(sel):
                    xb.append(float(np.mean(xvals[sel])))
                    yb.append(float(np.mean(yvals[sel])))
            if len(xb) >= 2:
                plt.plot(np.array(xb), np.array(yb), linewidth=3.0, label="binned means")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_legend:
        plt.legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _binned_within_plot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    q_bins: int = 25,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> None:
    """
    Paper-style binned within-network plot:
      - expects x,y already demeaned (within network)
      - trims x to [p_lo, p_hi] percentiles
      - bins by quantiles of x and plots bin means only
    """
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        raise ValueError("No finite rows for binned-within plot.")

    xvals = d[x].to_numpy()
    xlo, xhi = np.nanpercentile(xvals, [p_lo, p_hi])
    d = d[(d[x] >= xlo) & (d[x] <= xhi)].copy()
    if d.empty:
        raise ValueError("All rows trimmed out for binned-within plot.")

    d["bin"] = pd.qcut(d[x], q=q_bins, duplicates="drop")
    b = (
        d.groupby("bin", observed=True)
        .agg(
            x_mean=(x, "mean"),
            y_mean=(y, "mean"),
            n=(y, "size"),
        )
        .reset_index()
    )

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(b["x_mean"], b["y_mean"], s=60, label="bin means")
    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(0.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _hist_delta_vuln_core(
    core: pd.DataFrame,
    *,
    col: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    mode: str = "pos",  # "pos" | "nonzero" | "all"
) -> None:
    """
    Histogram of core-node vulnerability gaps with presentation-friendly conditioning.

    mode:
      - "pos":    plot Δν > 0 only (recommended), but annotate shares of <0,=0,>0 in full sample
      - "nonzero": plot Δν != 0 (shows both tails)
      - "all":    plot all values (will show spike at 0 if mass exists)
    """
    x_all = pd.to_numeric(core[col], errors="coerce").to_numpy()
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size == 0:
        raise ValueError(f"No finite data for {col}")

    # shares in the FULL sample
    #share_neg = float(np.mean(x_all < 0))
    #share_zero = float(np.mean(x_all == ZERO_TOL))
    #share_pos = float(np.mean(x_all > ZERO_TOL))

    shares = _shares_three_way(x_all, zero_tol=ZERO_TOL)
    share_neg = shares["share_neg"]
    share_zero = shares["share_zero"]
    share_pos = shares["share_pos"]

    # choose what to plot
    mode = str(mode).lower().strip()
    if mode == "pos":
        x = x_all[x_all > ZERO_TOL]
        subtitle = r" (conditional on $\Delta \nu > 0$)"
    elif mode == "nonzero":
        x = x_all[np.abs(x_all) != ZERO_TOL]
        subtitle = r" (conditional on $\Delta \nu \neq 0$)"
    elif mode == "all":
        x = x_all
        subtitle = ""
    else:
        raise ValueError("mode must be one of: 'pos', 'nonzero', 'all'")

    if x.size == 0:
        raise ValueError(f"No observations in requested histogram mode='{mode}'")

    # trim tails of the PLOTTED sample only (for readability)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    xt = x[(x >= lo) & (x <= hi)]
    if xt.size == 0:
        raise ValueError("All rows trimmed out for histogram (try wider p_lo/p_hi).")

    # stats for the PLOTTED sample
    mean = float(np.mean(xt))
    med = float(np.median(xt))

    fig = plt.figure()
    plt.hist(xt, bins=bins, density=True)

    # reference line at 0 (still useful even when plotting only positive tail)
    plt.axvline(0.0, linewidth=1.5)
    plt.axvline(med, linewidth=1.5)

    plt.title(title + subtitle)
    plt.xlabel(xlabel)
    plt.ylabel("Density")

    txt = "\n".join(
        [
            f"N total (core) = {x_all.size}",
            f"Share(Δν<0) = {share_neg:.3f}",
            f"Share(Δν=0) = {share_zero:.3f}",
            f"Share(Δν>0) = {share_pos:.3f}",
            "",
            f"N plotted = {x.size}",
            f"Mean (plotted) = {mean:.4f}",
            f"Median (plotted) = {med:.4f}",
            f"Trim (plotted) = [{p_lo:.1f}, {p_hi:.1f}] pct",
        ]
    )
    plt.gca().text(
        0.02,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _fmt_num(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return ""
    return f"{float(x):.{digits}f}"


def _extract_term(model, term: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (coef, se, pval) for a term, or (None,None,None) if absent.
    """
    if term not in model.params.index:
        return None, None, None
    coef = float(model.params[term])
    se = float(model.bse[term])
    p = float(model.pvalues[term])
    return coef, se, p


def _latex_reg_table(
    *,
    models: List,
    col_titles: List[str],
    n_obs: int,
    n_clusters: int,
    r2s: List[float],
    include_fe: bool = True,
    hhi_label: str = r"$\Delta \mathrm{HHI}^{\mathrm{out}}_{ig} / \mathrm{HHI}^{\mathrm{out,BFF}}_{ig}$",
    main_label: str = r"$\Delta s_{ig} / \ell^{\mathrm{BFF}}_{ig}$",
) -> str:
    """
    Build a 3-col LaTeX table (booktabs + tabularx) that matches your Overleaf style.
    """
    # row spec: (display_label, term_name)
    rows = [
        (main_label, "delta_slack_frac"),
        (r"$\log(1+\ell^{\mathrm{BFF}}_{ig})$", "log_ell_bff"),
        (r"$\log(1+\mathrm{in}^{\mathrm{unc}}_{ig})$", "log_in_unc"),
        (hhi_label, "delta_hhi_out_frac"),
    ]

    # format coefficient & se lines
    def cell_coef_se(model, term: str) -> Tuple[str, str]:
        coef, se, p = _extract_term(model, term)
        if coef is None:
            return "", ""
        c = _fmt_num(coef, 3) + _stars(p)
        s = "(" + _fmt_num(se, 3) + ")"
        return c, s

    # build body
    header = " & " + " & ".join(col_titles) + " \\\\"
    lines = []
    lines.append("% -------------------------------------------------------------------")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Core-node panel regressions: liquidity slack differences explain vulnerability differences}")
    lines.append("\\label{tab:exp3_slack_reg}")
    lines.append("\\begin{tabularx}{\\textwidth}{l" + "c" * len(models) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    # regressor rows: coef line + se line + spacing
    for label, term in rows:
        coef_line = [label]
        se_line = [""]
        any_present = False
        for m in models:
            c, s = cell_coef_se(m, term)
            coef_line.append(c)
            se_line.append(s)
            any_present = any_present or (c != "")
        if any_present:
            lines.append(" & ".join(coef_line) + " \\\\")
            lines.append(" & ".join(se_line) + " \\\\")
            lines.append("\\\\[-2pt]")

    lines.append("\\midrule")
    lines.append("Network fixed effects & " + " & ".join(["Yes" if include_fe else "No"] * len(models)) + " \\\\")
    lines.append("SE clustered by network & " + " & ".join(["Yes"] * len(models)) + " \\\\")
    lines.append(f"Observations (core nodes) & " + " & ".join([str(n_obs)] * len(models)) + " \\\\")
    lines.append(f"Networks (clusters) & " + " & ".join([str(n_clusters)] * len(models)) + " \\\\")
    lines.append("$R^2$ & " + " & ".join([_fmt_num(r, 3) for r in r2s]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\vspace{6pt}")
    lines.append("\\footnotesize")
    lines.append(
        "\\emph{Notes:} The dependent variable is the core-node vulnerability gap "
        "$\\Delta \\nu_{ig} = \\nu^{\\mathrm{maxC}}_{ig} - \\nu^{\\mathrm{BFF}}_{ig}$ under the common buffer scale "
        "$\\alpha=\\alpha^{\\ast}_{\\mathrm{BFF}}$. The key regressor is the slack difference normalised by outgoing "
        "obligations under BFF, $\\Delta s_{ig}/\\ell^{\\mathrm{BFF}}_{ig}$. "
        "All specifications include network fixed effects (implemented via within-network demeaning). "
        "Standard errors (in parentheses) are clustered at the network level. "
        "$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.1$."
    )
    lines.append("\\end{table}")
    lines.append("% -------------------------------------------------------------------")
    return "\n".join(lines)


def make_all(artifact_dir: Path) -> List[Path]:
    """
    Expects:
      artifact_dir/node_panel.csv

    Writes:
      artifact_dir/figures/...
      artifact_dir/tables/...

    Returns list of created file paths.
    """

    artifact_dir = Path(artifact_dir)
    node_path = artifact_dir / "node_panel.csv"
    if not node_path.exists():
        raise FileNotFoundError(f"Missing {node_path}")

    df = pd.read_csv(node_path)

    required = [
        "net_id",
        "tier",
        "delta_vuln_common",
        "delta_slack_common",
        "ell_bff",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"node_panel.csv missing columns: {missing}")

    core = df[df["tier"] == "core"].copy()
    core = core.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["delta_vuln_common", "delta_slack_common", "ell_bff", "net_id"]
    )

    core = _prepare_core(core, eps=1e-12)

    run_id = artifact_dir.name
    fig_dir = FIG_ROOT / "exp3" / run_id
    tab_dir = TABLE_ROOT / "exp3" / run_id
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    created: List[Path] = []

    # -------------------------------------------------
    # 1) Raw scatter: Δν vs Δs/ℓ (with legend)
    # -------------------------------------------------
    scatter_png = fig_dir / "exp3_scatter_core_delta_vuln_vs_delta_slack_frac.png"
    scatter_pdf = fig_dir / "exp3_scatter_core_delta_vuln_vs_delta_slack_frac.pdf"

    _scatter_with_fit(
        core,
        x="delta_slack_frac",
        y="delta_vuln_common",
        out_png=scatter_png,
        out_pdf=scatter_pdf,
        title="Core nodes: vulnerability gap vs normalized slack gap (common α = α_BFF*)",
        xlabel="Δ slack / ℓ (maxC − BFF) under common α",
        ylabel="Δ vulnerability (maxC − BFF) under common α",
        add_binned_means=True,
        n_bins=20,
        show_legend=True,
    )
    created += [scatter_png, scatter_pdf]

    # -------------------------------------------------
    # 2) Binned within-network (demeaned) plot (matches old “nice” style)
    # -------------------------------------------------
    scatter_dm_png = fig_dir / "exp3_binned_within_delta_vuln_vs_delta_slack_frac_core.png"
    scatter_dm_pdf = fig_dir / "exp3_binned_within_delta_vuln_vs_delta_slack_frac_core.pdf"

    _binned_within_plot(
        core,
        x="delta_slack_frac_dm",
        y="delta_vuln_common_dm",
        out_png=scatter_dm_png,
        out_pdf=scatter_dm_pdf,
        title="Binned within-network relationship (core nodes)",
        xlabel=r"Within-network demeaned normalised slack gap: $\Delta s/\ell$",
        ylabel=r"Within-network demeaned vulnerability gap: $\Delta \nu$",
        q_bins=25,
        p_lo=1.0,
        p_hi=99.0,
    )
    created += [scatter_dm_png, scatter_dm_pdf]

    # -------------------------------------------------
    # 2.5) Distribution: core-node vulnerability gap Δν
    # -------------------------------------------------
    hist_png = fig_dir / "exp3_hist_core_delta_vuln_common.png"
    hist_pdf = fig_dir / "exp3_hist_core_delta_vuln_common.pdf"

    _hist_delta_vuln_core(
        core,
        col="delta_vuln_common",
        out_png=hist_png,
        out_pdf=hist_pdf,
        title="Core nodes: distribution of vulnerability gap (maxC − BFF)",
        xlabel=r"$\Delta \nu$  (failure prob gap under common $\alpha$)",
        bins=60,
        p_lo=0.5,
        p_hi=99.5,
        mode="pos",  # <-- key change: plot Δν>0 only
    )
    created += [hist_png, hist_pdf]

    # -------------------------------------------------
    # 3) Regressions (cluster by net_id): keep the full dump for diagnostics
    # -------------------------------------------------
    f0 = "delta_vuln_common ~ delta_slack_frac"

    controls = []
    # concentration regressor: prefer normalized if available
    if "delta_hhi_out_frac" in core.columns:
        controls.append("delta_hhi_out_frac")
    # (optional) reliance
    if "incoming_reliance_bff" in core.columns:
        controls.append("incoming_reliance_bff")
    # size controls
    controls.append("log_ell_bff")
    if "log_in_unc" in core.columns:
        controls.append("log_in_unc")

    f1 = f0 + (" + " + " + ".join(controls) if controls else "")
    f2 = f1 + " + C(net_id)"

    # within-transformed (demean all vars by net_id)
    within_cols = ["delta_vuln_common", "delta_slack_frac"] + [c for c in controls if c in core.columns]
    w = core[["net_id"] + within_cols].copy()
    for c in within_cols:
        w[c] = w[c] - w.groupby("net_id")[c].transform("mean")

    f_within = "delta_vuln_common ~ delta_slack_frac" + (
        " + " + " + ".join([c for c in controls if c in w.columns]) if controls else ""
    )

    m0 = _cluster_ols(f0, core, cluster_col="net_id")
    m1 = _cluster_ols(f1, core, cluster_col="net_id")
    m2 = _cluster_ols(f2, core, cluster_col="net_id")
    mW = _cluster_ols(f_within, w, cluster_col="net_id")

    reg_txt = tab_dir / "exp3_reg_core.txt"
    reg_tex = tab_dir / "exp3_reg_core.tex"

    txt = []
    txt.append("============================================")
    txt.append("Exp3 core regressions: delta_vuln_common")
    txt.append("Cluster-robust SEs by net_id")
    txt.append("============================================\n")

    txt.append("[m0] " + f0)
    txt.append(str(m0.summary()))
    txt.append("\n[m1] " + f1)
    txt.append(str(m1.summary()))
    txt.append("\n[m2] " + f2)
    txt.append(str(m2.summary()))
    txt.append("\n[mW] WITHIN (demeaned by net_id; FE-equivalent slopes)")
    txt.append("[mW] " + f_within)
    txt.append(str(mW.summary()))
    txt.append("")

    _save_text(reg_txt, "\n".join(txt))

    tex = []
    tex.append(_summary_to_tex(m0, f"m0: {f0}"))
    tex.append(_summary_to_tex(m1, f"m1: {f1}"))
    tex.append(_summary_to_tex(m2, f"m2: {f2}"))
    tex.append(_summary_to_tex(mW, f"mW (within): {f_within}"))
    _save_text(reg_tex, "\n".join(tex))

    created += [reg_txt, reg_tex]
    
    # -------------------------------------------------
    # 4) Paper-ready LaTeX regression table (3 columns, FE via within)
    # -------------------------------------------------
    # Build FE-equivalent (within) specs for the table:
    # (1) baseline
    # (2) + size
    # (3) + concentration
    table_terms = ["delta_vuln_common", "delta_slack_frac", "log_ell_bff"]
    if "log_in_unc" in core.columns:
        table_terms.append("log_in_unc")
    if "delta_hhi_out_frac" in core.columns:
        table_terms.append("delta_hhi_out_frac")

    wT = core[["net_id"] + table_terms].copy()
    for c in table_terms:
        wT[c] = wT[c] - wT.groupby("net_id")[c].transform("mean")

    fT1 = "delta_vuln_common ~ delta_slack_frac"
    fT2 = "delta_vuln_common ~ delta_slack_frac + log_ell_bff" + (" + log_in_unc" if "log_in_unc" in wT.columns else "")
    fT3 = fT2 + (" + delta_hhi_out_frac" if "delta_hhi_out_frac" in wT.columns else "")

    mT1 = _cluster_ols(fT1, wT, cluster_col="net_id")
    mT2 = _cluster_ols(fT2, wT, cluster_col="net_id")
    mT3 = _cluster_ols(fT3, wT, cluster_col="net_id")

    reg_table_tex = tab_dir / "exp3_reg_core_table.tex"
    table_tex = _latex_reg_table(
        models=[mT1, mT2, mT3],
        col_titles=["(1) Baseline", "(2) + Size", "(3) + Concentration"],
        n_obs=int(len(core)),
        n_clusters=int(core["net_id"].nunique()),
        r2s=[float(mT1.rsquared), float(mT2.rsquared), float(mT3.rsquared)],
        include_fe=True,
    )
    _save_text(reg_table_tex, table_tex)
    created += [reg_table_tex]

    # -------------------------------------------------
    # 4.5) Machine-readable exp3 summary (for exporter)
    # -------------------------------------------------
    x_all = pd.to_numeric(core["delta_vuln_common"], errors="coerce").to_numpy()
    x_all = x_all[np.isfinite(x_all)]
    
    headlines = _shares_three_way(x_all, zero_tol=ZERO_TOL)
    headlines["n_networks"] = int(core["net_id"].nunique())

    terms = ["delta_slack_frac", "log_ell_bff", "log_in_unc", "delta_hhi_out_frac"]
    
    summary = {
        "run_id": run_id,
        "headlines": headlines,
        "reg_table": {
            "n_clusters": int(core["net_id"].nunique()),
            "models": [
                _model_summary(mT1, name="(1) Baseline", formula=fT1, terms=terms),
                _model_summary(mT2, name="(2) + Size", formula=fT2, terms=terms),
                _model_summary(mT3, name="(3) + Concentration", formula=fT3, terms=terms),
            ],
        },
    }
    
    summary_json = tab_dir / "exp3_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    # canonical copy for exporters / paper build
    summary_json_canon = PAPER_OUTPUT_ROOT / "exp3_summary.json"
    summary_json_canon.parent.mkdir(parents=True, exist_ok=True)
    summary_json_canon.write_text(json.dumps(summary, indent=2))
    
    created += [summary_json, summary_json_canon]
    
    #created += [summary_json]

    return created

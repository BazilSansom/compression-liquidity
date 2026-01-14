from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

from paper.io import load_json


def _repo_root() -> Path:
    # paper/build/export_paper_numbers.py -> build -> paper -> repo root
    return Path(__file__).resolve().parents[2]

def _canonical_exp3_summary() -> Path:
    try:
        from paper.paths import PAPER_OUTPUT_ROOT  # type: ignore
        return Path(PAPER_OUTPUT_ROOT) / "exp3_summary.json"
    except Exception:
        return _repo_root() / "outputs" / "paper" / "exp3_summary.json"


def _default_out_tex() -> Path:
    try:
        from paper.paths import PAPER_OUTPUT_ROOT
        return Path(PAPER_OUTPUT_ROOT) / "paper_numbers.tex"
    except Exception:
        return _repo_root() / "outputs" / "paper" / "paper_numbers.tex"


def _find_latest_exp3_summary(table_root: Path) -> Path:
    candidates = list(table_root.glob("exp3/*/exp3_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No exp3_summary.json files found under: {table_root / 'exp3'}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _fmt_float(x: Any, digits: int = 3) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return ""
    if xf != xf:  # NaN
        return ""
    # Use fixed-point (stable for LaTeX + diffs)
    return f"{xf:.{digits}f}"


def _fmt_int(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(int(x))
    except Exception:
        return ""


#def _cmd(name: str, value: str) -> str:
    # Always define the macro (even if value empty) to avoid LaTeX undefined errors
#    return f"\\newcommand{{\\{name}}}{{{value}}}"

def _cmd(name: str, value: str) -> str:
    return "\n".join([
        f"\\providecommand{{\\{name}}}{{}}",
        f"\\renewcommand{{\\{name}}}{{{value}}}",
    ])


def _get_model(summary: dict[str, Any], idx: int) -> dict[str, Any]:
    models = summary.get("reg_table", {}).get("models", [])
    if idx >= len(models):
        return {}
    return models[idx] or {}


def _get_term(model: dict[str, Any], term: str) -> dict[str, Any]:
    return (model.get("terms", {}) or {}).get(term, {}) or {}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_id",
        default=None,
        help="Run id folder name under TABLE_ROOT/exp3/<run_id>/exp3_summary.json. "
             "If omitted, uses the most recently modified exp3_summary.json.",
    )
    ap.add_argument(
        "--summary_json",
        default=None,
        help="Explicit path to an exp3_summary.json (overrides --run_id).",
    )
    ap.add_argument(
        "--out_tex",
        default=str(_default_out_tex()),
        help="Output .tex file for LaTeX macros (default: outputs/paper/paper_numbers.tex).",
    )
    ap.add_argument(
        "--digits_shares",
        type=int,
        default=3,
        help="Decimal places for share macros.",
    )
    ap.add_argument(
        "--digits_beta",
        type=int,
        default=3,
        help="Decimal places for regression coefficient/SE macros.",
    )
    ap.add_argument(
        "--digits_r2",
        type=int,
        default=3,
        help="Decimal places for R^2 macros.",
    )
    args = ap.parse_args(argv)

    # Locate TABLE_ROOT robustly
    try:
        from paper.paths import TABLE_ROOT  # type: ignore
        table_root = Path(TABLE_ROOT)
    except Exception:
        # Fallback: repo_root/outputs/paper/tables
        table_root = _repo_root() / "outputs" / "paper" / "tables"

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
    else:
        #if args.run_id:
        #    summary_path = table_root / "exp3" / args.run_id / "exp3_summary.json"
       # else:
       #    summary_path = _find_latest_exp3_summary(table_root)
        if args.run_id:
            summary_path = table_root / "exp3" / args.run_id / "exp3_summary.json"
        else:
            # First choice: canonical copy written by exp3_make_all
            canon = _canonical_exp3_summary()
            summary_path = canon if canon.exists() else _find_latest_exp3_summary(table_root)


    if not summary_path.exists():
        raise FileNotFoundError(f"Missing exp3_summary.json at: {summary_path}")

    summary = load_json(summary_path)

    headlines = summary.get("headlines", {}) or {}
    run_id = summary.get("run_id", "") or summary_path.parent.name

    # Headline shares (tolerance-aware; produced by exp3_make_all.py)
    zero_tol = headlines.get("zero_tol", None)
    n_core = headlines.get("n", None)
    n_networks = headlines.get("n_networks", None)
    share_neg = headlines.get("share_neg", None)
    share_zero = headlines.get("share_zero", None)
    share_pos = headlines.get("share_pos", None)

    # Regression models (table ones: (1) baseline, (2) + size, (3) + concentration)
    m1 = _get_model(summary, 0)
    m2 = _get_model(summary, 1)
    m3 = _get_model(summary, 2)

    term_main = "delta_slack_frac"

    t1 = _get_term(m1, term_main)
    t2 = _get_term(m2, term_main)
    t3 = _get_term(m3, term_main)

    # Build macros
    lines: list[str] = []
    lines.append("% =========================================================")
    lines.append("% Paper numbers (single source of truth)")
    lines.append("% AUTO-GENERATED by paper/build/export_paper_numbers.py")
    lines.append("% Source: " + str(summary_path))
    lines.append("% =========================================================\n")

    lines.append("% --- Exp3 headline shares (core nodes) ---")
    lines.append(_cmd("ExpThreeRunId", str(run_id)))
    lines.append(_cmd("ExpThreeZeroTol", _fmt_float(zero_tol, digits=2)))
    lines.append(_cmd("ExpThreeNCore", _fmt_int(n_core)))
    lines.append(_cmd("ExpThreeNNetworks", _fmt_int(n_networks)))

    lines.append(_cmd("ExpThreeShareNeg", _fmt_float(share_neg, digits=args.digits_shares)))
    lines.append(_cmd("ExpThreeShareZero", _fmt_float(share_zero, digits=args.digits_shares)))
    lines.append(_cmd("ExpThreeSharePos", _fmt_float(share_pos, digits=args.digits_shares)))

    # Percent versions (so you can write "\ExpThreeShareZeroPct\%")
    def pct(x: Any) -> Optional[float]:
        try:
            return 100.0 * float(x)
        except Exception:
            return None

    lines.append(_cmd("ExpThreeShareNegPct", _fmt_float(pct(share_neg), digits=args.digits_shares)))
    lines.append(_cmd("ExpThreeShareZeroPct", _fmt_float(pct(share_zero), digits=args.digits_shares)))
    lines.append(_cmd("ExpThreeSharePosPct", _fmt_float(pct(share_pos), digits=args.digits_shares)))

    lines.append("\n% --- Exp3 regression: beta on normalised slack gap ---")
    # (1)
    lines.append(_cmd("ExpThreeBetaSlackB", _fmt_float(t1.get("coef"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackBSE", _fmt_float(t1.get("se"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackBP", _fmt_float(t1.get("p"), digits=4)))
    lines.append(_cmd("ExpThreeRtwoB", _fmt_float(m1.get("rsquared"), digits=args.digits_r2)))
    # (2)
    lines.append(_cmd("ExpThreeBetaSlackS", _fmt_float(t2.get("coef"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackSSE", _fmt_float(t2.get("se"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackSP", _fmt_float(t2.get("p"), digits=4)))
    lines.append(_cmd("ExpThreeRtwoS", _fmt_float(m2.get("rsquared"), digits=args.digits_r2)))
    # (3)
    lines.append(_cmd("ExpThreeBetaSlackC", _fmt_float(t3.get("coef"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackCSE", _fmt_float(t3.get("se"), digits=args.digits_beta)))
    lines.append(_cmd("ExpThreeBetaSlackCP", _fmt_float(t3.get("p"), digits=4)))
    lines.append(_cmd("ExpThreeRtwoC", _fmt_float(m3.get("rsquared"), digits=args.digits_r2)))

    out_tex = Path(args.out_tex).expanduser()
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")

    print(f"[export_paper_numbers] Wrote: {out_tex}")
    print(f"[export_paper_numbers] From:  {summary_path}")


if __name__ == "__main__":
    main()

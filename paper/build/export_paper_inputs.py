# paper/build/export_paper_inputs.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from paper.specs.paper_base import PAPER_BASE, PAPER_MASTER_SEED, PAPER_N_DRAWS
from paper.specs.exp1 import SPEC as EXP1_SPEC  # Exp1Config
from paper.specs.exp2 import SPEC as EXP2_SPEC  # Exp2Config
from paper.specs.exp3 import SPEC as EXP3_SPEC  # Exp3Config

from paper.paths import PAPER_OUTPUT_ROOT
DEFAULT_OUT = PAPER_OUTPUT_ROOT / "paper_inputs.tex"


#DEFAULT_OUT = Path("outputs") / "paper" / "paper_inputs.tex"

_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPE_MAP.get(ch, ch) for ch in s)


def _fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (tuple, list)):
        return ", ".join(_fmt_value(x) for x in v)
    return _tex_escape(str(v))


def _emit_macro(name: str, value: Any, comment: str | None = None) -> str:
    """
    Robust macro emission:
      - safe if the file is accidentally \\input twice
      - safe if some macros are pre-defined elsewhere
    """
    val = _fmt_value(value)
    lines: list[str] = []
    if comment:
        lines.append(f"% {comment}")
    lines.append(f"\\providecommand{{\\{name}}}{{}}")
    lines.append(f"\\renewcommand{{\\{name}}}{{{val}}}")
    return "\n".join(lines)


def _export_base(lines: list[str]) -> None:
    lines.append("% --- Paper-level ---")
    lines.append(_emit_macro("PaperMasterSeed", PAPER_MASTER_SEED))
    lines.append(_emit_macro("PaperNDraws", PAPER_N_DRAWS))

    n = PAPER_BASE.network
    lines.append("\n% --- Network spec ---")
    lines.append(_emit_macro("NetNCore", n.n_core))
    lines.append(_emit_macro("NetNSource", n.n_source))
    lines.append(_emit_macro("NetNSink", n.n_sink))
    lines.append(_emit_macro("NetP", n.p))
    lines.append(_emit_macro("NetWeightMode", n.weight_mode))
    lines.append(_emit_macro("NetAlphaWeights", n.alpha_weights))
    lines.append(_emit_macro("NetScaleWeights", n.scale_weights))
    lines.append(_emit_macro("NetDegreeMode", n.degree_mode))
    lines.append(_emit_macro("NetRoundTo", n.round_to))
    lines.append(_emit_macro("NetUseLCC", n.use_lcc))

    sh = PAPER_BASE.shock
    lines.append("\n% --- Shock spec (base) ---")
    lines.append(_emit_macro("ShockRhoXi", sh.rho_xi))
    lines.append(_emit_macro("ShockXiScale", sh.xi_scale))
    lines.append(_emit_macro("ShockLamDefault", sh.lam_default))
    lines.append(_emit_macro("ShockLamGrid", sh.lam_grid))

    comp = PAPER_BASE.compression
    lines.append("\n% --- Compression spec (base) ---")
    lines.append(_emit_macro("CompMethodBase", comp.method))
    lines.append(_emit_macro("CompSolverBase", comp.solver))
    lines.append(_emit_macro("CompTolZero", comp.tol_zero))
    lines.append(_emit_macro("CompRequireConservative", comp.require_conservative))
    lines.append(_emit_macro("CompRequireFullConservative", comp.require_full_conservative))

    srch = PAPER_BASE.search
    lines.append("\n% --- Buffer search spec (base) ---")
    lines.append(_emit_macro("SearchTargetShortfall", srch.target_shortfall))
    lines.append(_emit_macro("SearchAlphaLo", srch.alpha_lo))
    lines.append(_emit_macro("SearchAlphaHi", srch.alpha_hi))
    lines.append(_emit_macro("SearchGrowFactor", srch.grow_factor))
    lines.append(_emit_macro("SearchMaxGrowSteps", srch.max_grow_steps))
    lines.append(_emit_macro("SearchTol", srch.tol))
    lines.append(_emit_macro("SearchMaxIter", srch.max_iter))
    lines.append(_emit_macro("SearchRelTolFPA", srch.rel_tol_fpa))


def _export_exp1(lines: list[str]) -> None:
    exp1 = EXP1_SPEC
    sp = exp1.spec

    lines.append("\n% =========================================================")
    lines.append("% Experiment 1 parameters")
    lines.append("% =========================================================")

    lines.append("\n% --- Exp1 config ---")
    lines.append(_emit_macro("ExpOneBufferTheta", exp1.buffer_theta))
    lines.append(_emit_macro("ExpOneMethods", exp1.methods, comment="compression methods compared"))
    lines.append(_emit_macro("ExpOneMaxCSolver", exp1.maxc_solver))

    sh = sp.shock
    lines.append("\n% --- Exp1 shock settings ---")
    lines.append(_emit_macro("ExpOneLamDefault", sh.lam_default))
    lines.append(_emit_macro("ExpOneLamGrid", sh.lam_grid))


def _export_exp2(lines: list[str]) -> None:
    exp2 = EXP2_SPEC
    sp = exp2.spec

    lines.append("\n% =========================================================")
    lines.append("% Experiment 2 parameters")
    lines.append("% =========================================================")

    lines.append("\n% --- Exp2 config ---")
    lines.append(_emit_macro("ExpTwoMethods", exp2.methods, comment="compression methods compared"))
    lines.append(_emit_macro("ExpTwoMaxCSolver", exp2.maxc_solver))

    # Optional but often nice: explicitly export the lambda settings used in exp2,
    # even if inherited from base, so the paper can cite Exp2LamRef etc if needed.
    sh = sp.shock
    lines.append("\n% --- Exp2 shock settings ---")
    lines.append(_emit_macro("ExpTwoLamDefault", sh.lam_default))
    lines.append(_emit_macro("ExpTwoLamGrid", sh.lam_grid))


def _export_exp3(lines: list[str]) -> None:
    exp3 = EXP3_SPEC
    sp = exp3.spec

    lines.append("\n% =========================================================")
    lines.append("% Experiment 3 parameters")
    lines.append("% =========================================================")

    lines.append("\n% --- Exp3 config ---")
    lines.append(_emit_macro("ExpThreeMaxCSolver", exp3.maxc_solver))
    lines.append(_emit_macro("ExpThreeLamRef", exp3.lam_ref))
    lines.append(_emit_macro("ExpThreeLamEval", exp3.lam_eval))
    lines.append(_emit_macro("ExpThreeNCalib", exp3.n_calib))
    lines.append(_emit_macro("ExpThreeAlphaAgg", exp3.alpha_agg))
    lines.append(_emit_macro("ExpThreeNXiDraws", exp3.n_xi_draws))
    lines.append(_emit_macro("ExpThreeTopK", exp3.topk))
    lines.append(_emit_macro("ExpThreeFailTol", exp3.fail_tol))

    # Sometimes useful to cite the underlying shock correlation / scale too:
    sh = sp.shock
    lines.append("\n% --- Exp3 shock settings (from ExperimentSpec) ---")
    lines.append(_emit_macro("ExpThreeRhoXi", sh.rho_xi))
    lines.append(_emit_macro("ExpThreeXiScale", sh.xi_scale))


def export(out_path: Path) -> Path:
    out_path = Path(out_path)

    lines: list[str] = []
    lines.append("% =========================================================")
    lines.append("% Paper parameters (single source of truth)")
    lines.append("% AUTO-GENERATED by paper/build/export_paper_inputs.py")
    lines.append("% =========================================================\n")

    _export_base(lines)
    _export_exp1(lines)
    _export_exp2(lines)
    _export_exp3(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .tex file (default: outputs/paper/paper_inputs.tex)",
    )
    args = ap.parse_args(argv)

    p = export(args.out)
    print(f"Wrote: {p}")


if __name__ == "__main__":
    main()

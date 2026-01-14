# paper/build/make_all.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from paper.specs.exp1 import SPEC as EXP1_SPEC, DEFAULT_SEEDS as EXP1_SEEDS
from paper.specs.exp2 import SPEC as EXP2_SPEC, DEFAULT_SEEDS as EXP2_SEEDS
from paper.specs.exp3 import SPEC as EXP3_SPEC, DEFAULT_SEEDS as EXP3_SEEDS

from paper.runners.exp1 import run_exp1
from paper.runners.exp2 import run_exp2
from paper.runners.exp3 import run_exp3

from paper.figures.exp1_make_all import make_all as exp1_make_all
from paper.figures.exp2_make_all import make_all as exp2_make_all
from paper.figures.exp3_make_all import make_all as exp3_make_all

from paper.build.export_paper_inputs import main as export_paper_inputs
from paper.build.export_paper_numbers import main as export_paper_numbers


def _pick_seeds(seeds: Sequence[int], mode: str) -> Sequence[int]:
    if mode == "full":
        return seeds
    # "small" smoke run: one seed only
    return seeds[:2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce all paper artifacts (exp1–exp3 + exporters).")
    ap.add_argument("--mode", choices=["small", "full"], default="small",
                    help="small: 1-seed smoke run; full: DEFAULT_SEEDS.")
    ap.add_argument("--tag", default="paper", help="Run tag embedded in output paths / metadata.")
    ap.add_argument("--skip-figures", action="store_true", help="Run sims only (no figure/table builds).")
    ap.add_argument("--skip-export", action="store_true", help="Skip TeX exporters.")
    ap.add_argument("--skip-exp3-figures", action="store_true", help="Skip exp3 figure/table build.")
    args = ap.parse_args()

    # exp1
    s1 = _pick_seeds(EXP1_SEEDS, args.mode)
    res1 = run_exp1(EXP1_SPEC, draw_ids=list(s1), tag=args.tag)
    a1 = Path(res1.artifact_dir)
    print(f"[exp1] Artifacts: {a1}")
    if not args.skip_figures:
        exp1_make_all(a1)

    # exp2
    s2 = _pick_seeds(EXP2_SEEDS, args.mode)
    res2 = run_exp2(EXP2_SPEC, list(s2), tag=args.tag)  # exp2 runner signature differs
    a2 = Path(res2.artifact_dir)
    print(f"[exp2] Artifacts: {a2}")
    if not args.skip_figures:
        exp2_make_all(a2)

    # exp3
    s3 = _pick_seeds(EXP3_SEEDS, args.mode)
    res3 = run_exp3(EXP3_SPEC, draw_ids=list(s3), tag=args.tag)
    a3 = Path(res3.artifact_dir)
    print(f"[exp3] Artifacts: {a3}")
    if (not args.skip_figures) and (not args.skip_exp3_figures):
        try:
            exp3_make_all(a3)
        except Exception as e:
            # Keep make_all robust: exp3 figs optional
            print(f"[exp3] make_all skipped/failed: {e}")

    # exporters (run after exp3 so exp3_summary.json exists)
    if not args.skip_export:
        export_paper_inputs(argv=[])
        export_paper_numbers(argv=[])


    print("[make_all] Done.")


if __name__ == "__main__":
    main()

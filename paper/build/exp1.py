#paper/build/exp1.py
from __future__ import annotations

from pathlib import Path

from paper.specs.exp1 import SPEC, DEFAULT_SEEDS
from paper.runners.exp1 import run_exp1
from paper.figures.exp1_make_all import make_all


def main() -> None:
    res = run_exp1(SPEC, draw_ids=DEFAULT_SEEDS, tag="paper")
    make_all(Path(res.artifact_dir))
    print(f"Artifacts: {res.artifact_dir}")


if __name__ == "__main__":
    main()

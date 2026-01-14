#paper/build/exp2.py
from __future__ import annotations

from pathlib import Path

from paper.specs.exp2 import SPEC, DEFAULT_SEEDS
from paper.runners.exp2 import run_exp2
from paper.figures.exp2_make_all import make_all


def main() -> None:
    res = run_exp2(SPEC, DEFAULT_SEEDS, tag="paper")
    make_all(Path(res.artifact_dir))
    print(f"Artifacts: {res.artifact_dir}")


if __name__ == "__main__":
    main()

# paper/build/exp3.py
from __future__ import annotations

from pathlib import Path

from paper.specs.exp3 import SPEC, DEFAULT_SEEDS
from paper.runners.exp3 import run_exp3

# optional; safe to omit if you haven't implemented figures yet
from paper.figures.exp3_make_all import make_all


def main() -> None:
    res = run_exp3(
        SPEC,
        draw_ids=DEFAULT_SEEDS,
        tag="paper",
    )

    # optional figure build
    try:
        make_all(Path(res.artifact_dir))
    except Exception as e:
        print(f"[exp3] make_all skipped/failed: {e}")

    print(f"Artifacts: {res.artifact_dir}")


if __name__ == "__main__":
    main()

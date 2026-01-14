# paper/figures/exp1_make_all.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from paper.figures.exp1_normalized_shortfall_reduction import make as make_norm
from paper.figures.exp1_activation import make as make_act
from paper.figures.exp1_mechanism_shares import make as make_mech


def make_all(artifact_dir: Path) -> list[Path]:
    artifact_dir = Path(artifact_dir)
    outs: list[Path] = []
    outs.append(make_norm(artifact_dir))
    outs.append(make_act(artifact_dir))
    outs.append(make_mech(artifact_dir))
    return outs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, help="Path to outputs/paper/artifacts/exp1/<run_id>")
    args = ap.parse_args(argv)

    outs = make_all(Path(args.artifact_dir))
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()

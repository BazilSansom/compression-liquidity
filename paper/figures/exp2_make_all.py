from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from paper.io import ensure_dir
from paper.paths import FIG_ROOT

from src.plots import plot_overlay_vs_lambda


def make_all(artifact_dir: Path) -> list[Path]:
    artifact_dir = Path(artifact_dir)
    run_id = artifact_dir.name

    out_dir = FIG_ROOT / "exp2" / run_id
    ensure_dir(out_dir)

    draws = pd.read_csv(artifact_dir / "draws.csv")
    rows = draws.to_dict(orient="records")

    outs: list[Path] = []

    out1 = out_dir / "exp2_erls_vs_lambda.png"
    plot_overlay_vs_lambda(
        rows,
        group_key="compression_method",
        y_key="erls",
        y_label="ERLS",
        hline=0.0,
        title=r"Equal-risk liquidity savings vs $\lambda$",
        out_png=out1,
    )
    outs.append(out1)

    out2 = out_dir / "exp2_kappa_vs_lambda.png"
    plot_overlay_vs_lambda(
        rows,
        group_key="compression_method",
        y_key="kappa",
        y_label=r"$\kappa$ (required buffer scale factor)",
        hline=1.0,
        title=r"Conservativeness ratio $\kappa$ vs $\lambda$",
        out_png=out2,
    )
    outs.append(out2)

    return outs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args(argv)

    for p in make_all(Path(args.artifact_dir)):
        print(p)


if __name__ == "__main__":
    main()

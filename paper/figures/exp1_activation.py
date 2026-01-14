from __future__ import annotations

from pathlib import Path
import pandas as pd

from paper.io import ensure_dir
from paper.paths import FIG_ROOT
from src.plots import plot_activation_pre_post


def make(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir)
    run_id = artifact_dir.name
    out_dir = FIG_ROOT / "exp1" / run_id
    ensure_dir(out_dir)

    draws = pd.read_csv(artifact_dir / "draws.csv")
    rows = draws.to_dict(orient="records")

    out_png = out_dir / "exp1_activation_share_pre_post_vs_lambda.png"
    plot_activation_pre_post(rows, out_png)
    return out_png


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args()
    print(make(Path(args.artifact_dir)))

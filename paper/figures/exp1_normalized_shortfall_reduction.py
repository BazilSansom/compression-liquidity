from __future__ import annotations

from pathlib import Path
import pandas as pd

from paper.io import ensure_dir
from paper.paths import FIG_ROOT
from src.plots import plot_overlay_vs_lambda


def make(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir)
    run_id = artifact_dir.name
    out_dir = FIG_ROOT / "exp1" / run_id
    ensure_dir(out_dir)

    draws = pd.read_csv(artifact_dir / "draws.csv")
    rows = draws.to_dict(orient="records")

    out_png = out_dir / "exp1_normalized_shortfall_reduction_vs_lambda.png"
    plot_overlay_vs_lambda(
        rows,
        group_key="compression_method",
        y_key="delta_R_norm",
        y_label=r"$(R_{\mathrm{pre}}-R_{\mathrm{post}})/\sum_{ij} V_{ij}$",
        hline=0.0,
        title=r"Normalized shortfall reduction vs $\lambda$",
        out_png=out_png,
    )
    return out_png


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args()
    print(make(Path(args.artifact_dir)))

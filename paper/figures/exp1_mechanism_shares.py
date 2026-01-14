from __future__ import annotations

from pathlib import Path
import pandas as pd

from paper.io import ensure_dir
from paper.paths import FIG_ROOT
from src.plots import plot_stacked_shares_vs_lambda


def make(artifact_dir: Path) -> Path:
    artifact_dir = Path(artifact_dir)
    run_id = artifact_dir.name
    out_dir = FIG_ROOT / "exp1" / run_id
    ensure_dir(out_dir)

    draws = pd.read_csv(artifact_dir / "draws.csv")
    rows = draws.to_dict(orient="records")

    out_png = out_dir / "exp1_theta_stacked_shares.png"
    plot_stacked_shares_vs_lambda(
        rows,
        group_key="compression_method",
        share_keys=(
            "share_new_trigger",
            "share_new_late",
            "share_still_inactive_cash_relief",
            "share_still_inactive_obligation_relief",
        ),
        share_labels=(
            r"$\Delta R_{\mathrm{new,trigger}}/\Delta R$",
            r"$\Delta R_{\mathrm{new,late}}/\Delta R$",
            r"$\Delta R_{\mathrm{cash\ relief\ on\ inactive}}/\Delta R$",
            r"$\Delta R_{\mathrm{obligation\ relief\ on\ inactive}}/\Delta R$",
        ),
        title="Exp1 fixed buffers | Mechanism shares",
        out_png=out_png,
        two_panel=True,
        show_n=True,
    )
    return out_png


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args()
    print(make(Path(args.artifact_dir)))

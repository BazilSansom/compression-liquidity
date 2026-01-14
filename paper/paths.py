# paper/paths.py
from __future__ import annotations

import os
from pathlib import Path


def _repo_root() -> Path:
    # repo_root / paper / paths.py  -> parents[1] is repo_root
    return Path(__file__).resolve().parents[1]


REPO_ROOT: Path = _repo_root()

# Allow override for CI / clusters, otherwise default to repo_root/outputs
OUTPUT_ROOT: Path = Path(os.getenv("OUTPUT_ROOT", REPO_ROOT / "outputs"))

PAPER_OUTPUT_ROOT: Path = OUTPUT_ROOT / "paper"
ARTIFACT_ROOT: Path = PAPER_OUTPUT_ROOT / "artifacts"
FIG_ROOT: Path = PAPER_OUTPUT_ROOT / "figures"
TABLE_ROOT: Path = PAPER_OUTPUT_ROOT / "tables"

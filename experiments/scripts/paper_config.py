# experiments/scripts/paper_config.py
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.base_cases import BASE_CASE

N_DRAWS: int = 20  # bump for paper
THETA_FIXED_BUFFERS: float = 1.0

FIG_ROOT = Path("figures") / "paper"

PAPER_BASE_SPEC = replace(BASE_CASE, name="paper_base")

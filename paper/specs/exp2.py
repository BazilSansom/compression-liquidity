# paper/specs/exp2.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

from paper.specs.paper_base import PAPER_BASE, PAPER_N_DRAWS
from src.experiment_specs import ExperimentSpec

# draw_ids for paper runs
DEFAULT_SEEDS: List[int] = list(range(PAPER_N_DRAWS))

EXP2_SPEC: ExperimentSpec = replace(
    PAPER_BASE,
    name="exp2",
    # Optionally override exp2-specific settings here, e.g.
    # shock=replace(PAPER_BASE.shock, lam_grid=(0.0, 0.25, 0.5, 0.75, 1.0)),
)

@dataclass(frozen=True)
class Exp2Config:
    spec: ExperimentSpec
    methods: Tuple[str, ...] = ("bff", "maxc")
    maxc_solver: str = "ortools"

SPEC = Exp2Config(spec=EXP2_SPEC)

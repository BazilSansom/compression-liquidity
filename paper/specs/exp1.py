from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple, List

from paper.specs.paper_base import PAPER_BASE, PAPER_N_DRAWS
from src.experiment_specs import ExperimentSpec

DEFAULT_SEEDS: List[int] = list(range(PAPER_N_DRAWS))

EXP1: ExperimentSpec = replace(
    PAPER_BASE,
    name="exp1",
    shock=replace(
        PAPER_BASE.shock,
        lam_grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5), # not different, just to demo replace
        lam_default=0.6,
    ),
)

@dataclass(frozen=True)
class Exp1Config:
    spec: ExperimentSpec
    buffer_theta: float = 1           # <- my THETA_FIXED_BUFFERS in original code
    methods: Tuple[str, ...] = ("bff", "maxc")
    maxc_solver: str = "ortools"
    #use_lcc: bool = True

SPEC = Exp1Config(spec=EXP1)

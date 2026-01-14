# paper/specs/exp3.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from paper.specs.paper_base import PAPER_BASE, PAPER_N_DRAWS
from src.experiment_specs import ExperimentSpec

# draw_ids for paper runs (controlled globally via PAPER_N_DRAWS)
DEFAULT_SEEDS: List[int] = list(range(PAPER_N_DRAWS))

# ExperimentSpec for exp3 (usually same as PAPER_BASE; override here if ever needed)
EXP3_SPEC: ExperimentSpec = replace(
    PAPER_BASE,
    name="exp3",
    # Optionally override exp3-specific settings here if desired.
    # e.g. shock=replace(PAPER_BASE.shock, rho_xi=0.2),
)

@dataclass(frozen=True)
class Exp3Config:
    """
    Experiment 3: vulnerability + slack under a common buffer calibration.

    Calibration:
      - draw n_calib shock shapes U (CRN across methods)
      - compute alpha_ref_bff and alpha_ref_maxc via ERLS-style root search
      - evaluate both methods at alpha_common = alpha_ref_bff

    Evaluation:
      - draw n_xi_draws shock shapes and estimate node vulnerability probabilities
      - compute mean slack under the same shock draws
    """
    spec: ExperimentSpec

    # compression solver choice for max-C
    maxc_solver: str = "ortools"

    # calibration + evaluation stress intensities
    lam_ref: float = 0.75
    lam_eval: float = 0.75

    # calibration and evaluation sample sizes
    n_calib: int = 100
    alpha_agg: str = "median"   # "median" or "mean"
    n_xi_draws: int = 100

    # diagnostics
    topk: int = 5
    fail_tol: float = 1e-12


SPEC = Exp3Config(spec=EXP3_SPEC)

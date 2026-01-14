
# paper/specs/paper_base.py
from src.experiment_specs import (
    ExperimentSpec, NetworkSpec, ShockSpec, CompressionSpec, BufferSearchSpec
)

# Paper-wide Monte Carlo default (not part of ExperimentSpec)
PAPER_MASTER_SEED=12345
PAPER_N_DRAWS = 20

PAPER_BASE = ExperimentSpec(
    name="base_spec",
    network=NetworkSpec(
        n_core=30, n_source=10, n_sink=10, p=0.4,
        weight_mode="pareto", alpha_weights=2.0, scale_weights=1.0,
        degree_mode="bernoulli", round_to=0.01,
        use_lcc=True,
    ),
    shock=ShockSpec(
        rho_xi=0,
        xi_scale="row_sum",
        lam_default=0.6,
        lam_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    ),
    compression=CompressionSpec(
        method="bff",
        solver="ortools",
        tol_zero=1e-12,
        require_conservative=True,
        require_full_conservative=True,
    ),
    search=BufferSearchSpec(
        target_shortfall=1e-10,   # or 1e-8 if our numbers are large
        alpha_lo=0.0,
        alpha_hi=4.0,
        grow_factor=2.0,
        max_grow_steps=50,
        tol=1e-6,
        max_iter=80,
        rel_tol_fpa=1e-8,
    ),
)




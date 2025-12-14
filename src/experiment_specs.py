from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import numpy as np

from src.buffers import behavioural_base_from_V


# --- Types ---
XiScale = Literal["row_sum", "col_sum", "total"]
BufferMode = Literal["fixed_shape", "behavioural"]


@dataclass(frozen=True)
class BufferSearchSpec:
    target_shortfall: float = 0.0
    alpha_lo: float = 0.0
    alpha_hi: float = 1.0
    grow_factor: float = 2.0
    max_grow_steps: int = 30
    tol: float = 1e-6
    max_iter: int = 60
    rel_tol_fpa: float = 1e-8


# For shocks/generate_uniform_factor_shapes
@dataclass(frozen=True)
class ShockSpec:
    """
    Shock config for Option A:
    - draw one U_base using generate_uniform_factor_shapes (rho_xi, seed_offset+draw_id)
    - reuse U_base across lam_grid (common random numbers)
    - xi(lam) = lam * scale(V_ref) * U_base
    """
    rho_xi: float = 0.3
    xi_scale: XiScale = "row_sum"
    # lam_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21))
    lam_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seed_offset: int = 10_000  # base offset so seeds don’t collide with network seeds
    u_mode: Literal["uniform_factor"] = "uniform_factor" # U_base reused across λ


@dataclass(frozen=True)
class NetworkSpec:
    # Core inputs
    n_core: int = 30
    n_source: int = 10
    n_sink: int = 10
    p: float = 0.4

    # Edge weights
    weight_mode: str = "pareto"
    alpha_weights: float = 2.0
    scale_weights: float = 1.0
    round_to: float | None = 0.01

    # Degree / topology options
    degree_mode: str = "bernoulli"

    # RNG control
    seed_offset: int = 1_000

    # Use largest connected component
    use_lcc: bool = True

    @property
    def N(self) -> int:
        return self.n_core + self.n_source + self.n_sink


@dataclass(frozen=True)
class CompressionSpec:
    method: Literal["bff", "maxc"] = "bff"
    solver: Literal["ortools", "pulp"] = "ortools"  # only used if method="maxc"
    tol_zero: float = 1e-12
    require_conservative: bool = True
    require_full_conservative: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    """
    One place to declare *everything* needed to run an experiment.
    """
    name: str
    network: NetworkSpec
    shock: ShockSpec
    compression: CompressionSpec
    buffer_mode: BufferMode = "behavioural"

    # Buffer shape rule (callable): base = buffer_shape_fn(V)
    buffer_shape_fn: Callable[[np.ndarray], np.ndarray] = behavioural_base_from_V

    # Buffer search configuration
    search: BufferSearchSpec = BufferSearchSpec()

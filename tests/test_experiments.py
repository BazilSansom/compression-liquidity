import numpy as np
import pytest

from src.experiments import find_min_buffers_for_target_shortfall
from src.simulation import run_fpa
from src.buffers import behavioural_base_from_V

from tests.fixtures import (
    bardoscia_example1,
    bardoscia_example2,
    bardoscia_example3,
)


def _as_col(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


@pytest.mark.integration  # this is more integration-ish than unit
@pytest.mark.parametrize(
    "fixture_fn",
    [bardoscia_example1, bardoscia_example2, bardoscia_example3],
)
def test_find_min_buffers_hits_target_and_is_near_minimal(fixture_fn):
    V, _e0, *_ = fixture_fn()  # <-- refactor for revised fixtures

    # Base buffer shape from behavioural rule (row sums of V)
    b_base = behavioural_base_from_V(V)

    # Deterministic liquidity shock: 20% of outgoing obligations
    lam = 0.2
    p = V.sum(axis=1)
    xi = lam * p

    target = 0.0

    alpha_star, b_star, fpa_star = find_min_buffers_for_target_shortfall(
        V=V,
        xi=xi,
        target_shortfall=target,
        b_base=b_base,  # fixed shape scaling
        alpha_lo=0.0,
        alpha_hi=1.0,
    )

    # 1) Feasibility: should meet the target
    assert fpa_star.aggregate_shortfall <= target + 1e-10

    # 2) Consistency check: rerun directly at returned buffers
    b0 = _as_col(b_star) - _as_col(xi)
    res_check = run_fpa(V, b0)
    assert res_check.aggregate_shortfall <= target + 1e-10

    # 3) Near-minimality: slightly smaller alpha should fail (unless alpha_star is ~0)
    if alpha_star > 1e-8:
        eps = 1e-3
        alpha_down = alpha_star * (1.0 - eps)
        b_down = alpha_down * _as_col(b_base)
        b0_down = b_down - _as_col(xi)
        res_down = run_fpa(V, b0_down)

        assert res_down.aggregate_shortfall > target + 1e-12

# tests/test_fpa_paper_examples.py
"""
Test run_fpa against canonical examples from Bardoscia et al. (2019).
These are deterministic, small (3 nodes), and should run fast.
"""

import numpy as np
import pytest

from src.simulation import run_fpa
from tests.fixtures import bardoscia_example1, bardoscia_example2, bardoscia_example3


def _as_col(x: np.ndarray) -> np.ndarray:
    """
    Coerce to (n, 1) column vector for shape-robust comparisons.
    Accepts (n,), (n,1), (1,n).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2 and a.shape[0] == 1 and a.shape[1] > 1:
        return a.reshape(-1, 1)
    return a


@pytest.mark.unit
@pytest.mark.parametrize(
    "example_fn",
    [bardoscia_example1, bardoscia_example2, bardoscia_example3],
)
def test_fpa_bardoscia_examples_match_expected_states(example_fn):
    L, e0, exp_residual, exp_cash, exp_shortfall, exp_agg_shortfall = example_fn()

    result = run_fpa(L, e0)

    # --- Required outputs exist ---
    assert hasattr(result, "residual_obligations")
    assert hasattr(result, "shortfall")
    assert hasattr(result, "aggregate_shortfall")

    # --- Shape-robust comparisons ---
    got_residual = _as_col(result.residual_obligations)
    got_shortfall = _as_col(result.shortfall)

    exp_residual = _as_col(exp_residual)
    exp_shortfall = _as_col(exp_shortfall)

    assert got_residual.shape == exp_residual.shape
    assert got_shortfall.shape == exp_shortfall.shape

    assert np.allclose(got_residual, exp_residual, atol=1e-12, rtol=0.0)
    assert np.allclose(got_shortfall, exp_shortfall, atol=1e-12, rtol=0.0)
    assert float(result.aggregate_shortfall) == pytest.approx(float(exp_agg_shortfall), abs=1e-12)

    # Only assert final_cash when the fixture provides it (examples 2 & 3).
    if exp_cash is not None:
        assert hasattr(result, "final_cash")
        got_cash = _as_col(result.final_cash)
        exp_cash = _as_col(exp_cash)
        assert got_cash.shape == exp_cash.shape
        assert np.allclose(got_cash, exp_cash, atol=1e-12, rtol=0.0)


@pytest.mark.unit
def test_fpa_example1_all_obligations_paid_is_equivalent_to_zero_shortfall_and_zero_residual():
    # A slightly stronger “semantic” test for example 1
    L, e0, _, _, _, _ = bardoscia_example1()
    result = run_fpa(L, e0)

    assert np.allclose(_as_col(result.residual_obligations), 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(_as_col(result.shortfall), 0.0, atol=1e-12, rtol=0.0)
    assert float(result.aggregate_shortfall) == pytest.approx(0.0, abs=1e-12)

@pytest.mark.unit
@pytest.mark.parametrize(
    "example_fn",
    [bardoscia_example1, bardoscia_example2, bardoscia_example3],
)
def test_fpa_nodes_that_pay_have_zero_residual(example_fn):
    L, e0, *_ = example_fn()
    res = run_fpa(L, e0)

    act = np.asarray(res.activation_round).reshape(-1)
    resid = _as_col(res.residual_obligations).reshape(-1)

    paid = act >= 1
    if np.any(paid):
        assert np.max(resid[paid]) <= 1e-12

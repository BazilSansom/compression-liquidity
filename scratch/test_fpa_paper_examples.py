""" Testing run_fpa against cannonical examples 
in original FAP paper (Bardocia et al 2019)"""

import numpy as np
import pytest

from compression.simulation import run_fpa


def _build_cycle_L(pbar):
    """
    Helper: build 3x3 liability matrix for the A->B->C->A cycle
    used in Bardoscia et al.'s FPA examples.
    """
    Pi = np.array([
        [0.0, 1.0, 0.0],  # A -> B
        [0.0, 0.0, 1.0],  # B -> C
        [1.0, 0.0, 0.0],  # C -> A
    ])
    return np.diag(pbar) @ Pi


def test_fpa_example1_all_shortfalls_zero():
    # Example 1 from Bardoscia et al. (Figure 1)
    # pbar(0) = (10, 7, 5), e(0) = (12, 4, 3)
    L = _build_cycle_L(np.array([10.0, 7.0, 5.0]))
    e0 = np.array([12.0, 4.0, 3.0])

    result = run_fpa(L, e0)

    # All obligations eventually paid
    assert np.allclose(result.residual_obligations, 0.0)

    # Shortfalls should be zero for all nodes
    assert np.allclose(result.shortfall, 0.0)
    assert result.aggregate_shortfall == 0.0


def test_fpa_example2_shortfall_only_at_C():
    # Example 2 from Bardoscia et al. (Figure 2)
    # pbar(0) = (10, 7, 12), e(0) = (12, 4, 3)
    L = _build_cycle_L(np.array([10.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])

    result = run_fpa(L, e0)

    # Expected steady-state obligations & cash from the paper:
    # pbar_ss = (0, 0, 12), e_ss = (7, 7, 10)
    expected_residual = np.array([[0.0], [0.0], [12.0]])
    expected_cash = np.array([[7.0], [7.0], [10.0]])
    expected_shortfall = np.array([[0.0], [0.0], [2.0]])

    assert np.allclose(result.residual_obligations, expected_residual)
    assert np.allclose(result.final_cash, expected_cash)
    assert np.allclose(result.shortfall, expected_shortfall)
    assert result.aggregate_shortfall == pytest.approx(2.0)


def test_fpa_example3_shortfalls_B_and_C():
    # Example 3 from Bardoscia et al. (Figure 3)
    # pbar(0) = (2, 7, 12), e(0) = (12, 4, 3)
    L = _build_cycle_L(np.array([2.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])

    result = run_fpa(L, e0)

    # Expected steady-state obligations & cash from the paper:
    # pbar_ss = (0, 7, 12), e_ss = (10, 6, 3)
    expected_residual = np.array([[0.0], [7.0], [12.0]])
    expected_cash = np.array([[10.0], [6.0], [3.0]])
    expected_shortfall = np.array([[0.0], [1.0], [9.0]])

    assert np.allclose(result.residual_obligations, expected_residual)
    assert np.allclose(result.final_cash, expected_cash)
    assert np.allclose(result.shortfall, expected_shortfall)
    assert result.aggregate_shortfall == pytest.approx(10.0)

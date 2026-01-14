import numpy as np
import pytest

from src.stats import gini
from src.stats import flow_hhi


 # Gini tests

def test_gini_zero_vector():
    x = np.zeros(10)
    assert gini(x) == 0.0


def test_gini_equal_values():
    x = np.ones(10)
    assert np.isclose(gini(x), 0.0)


def test_gini_single_nonzero():
    x = np.zeros(10)
    x[0] = 1.0
    # Maximal inequality: one node has everything
    g = gini(x)
    assert 0.85 < g < 1.0


def test_gini_scale_invariance():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(gini(x), gini(10.0 * x))


def test_gini_known_example():
    # Simple hand-checkable case
    x = np.array([0.0, 0.0, 1.0, 1.0])
    g = gini(x)
    # Known value = 0.5
    assert np.isclose(g, 0.5)



# HHI tests

def test_hhi_zero_for_no_flows():
    V = np.zeros((3, 3))
    hhi_in = flow_hhi(V, mode="incoming")
    hhi_out = flow_hhi(V, mode="outgoing")

    assert np.allclose(hhi_in, 0.0)
    assert np.allclose(hhi_out, 0.0)


def test_hhi_single_counterparty():
    # Node 1 receives everything from node 0
    V = np.array([
        [0.0, 10.0],
        [0.0,  0.0],
    ])

    hhi_in = flow_hhi(V, mode="incoming")
    hhi_out = flow_hhi(V, mode="outgoing")

    # Incoming: node 1 fully concentrated
    assert hhi_in[1] == pytest.approx(1.0)
    assert hhi_in[0] == pytest.approx(0.0)

    # Outgoing: node 0 fully concentrated
    assert hhi_out[0] == pytest.approx(1.0)
    assert hhi_out[1] == pytest.approx(0.0)


def test_hhi_two_equal_counterparties():
    # Node 2 receives equal flows from nodes 0 and 1
    V = np.array([
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
    ])

    hhi_in = flow_hhi(V, mode="incoming")

    # Two equal shares -> HHI = 0.5
    assert hhi_in[2] == pytest.approx(0.5)
    assert hhi_in[0] == pytest.approx(0.0)
    assert hhi_in[1] == pytest.approx(0.0)


def test_hhi_scale_invariant():
    V = np.array([
        [0.0, 2.0],
        [0.0, 0.0],
    ])

    hhi1 = flow_hhi(V, mode="incoming")
    hhi2 = flow_hhi(10.0 * V, mode="incoming")

    assert np.allclose(hhi1, hhi2)


def test_hhi_bounds():
    rng = np.random.default_rng(0)
    V = rng.random((5, 5))

    hhi_in = flow_hhi(V, mode="incoming")
    hhi_out = flow_hhi(V, mode="outgoing")

    assert np.all(hhi_in >= 0.0)
    assert np.all(hhi_in <= 1.0)
    assert np.all(hhi_out >= 0.0)
    assert np.all(hhi_out <= 1.0)


def test_invalid_mode_raises():
    V = np.zeros((2, 2))
    with pytest.raises(ValueError):
        flow_hhi(V, mode="sideways")


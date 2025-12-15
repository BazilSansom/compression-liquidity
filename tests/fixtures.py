# tests/fixtures.py
import numpy as np


def _build_cycle_L(pbar: np.ndarray) -> np.ndarray:
    """
    3x3 liability matrix for the A->B->C->A cycle used in Bardoscia et al. (2019).
    pbar[i] is total obligations of node i.
    """
    Pi = np.array(
        [
            [0.0, 1.0, 0.0],  # A -> B
            [0.0, 0.0, 1.0],  # B -> C
            [1.0, 0.0, 0.0],  # C -> A
        ],
        dtype=float,
    )
    return np.diag(np.asarray(pbar, dtype=float)) @ Pi


def bardoscia_example1():
    # Figure 1: pbar(0)=(10,7,5), e(0)=(12,4,3)
    L = _build_cycle_L(np.array([10.0, 7.0, 5.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    expected_residual = np.zeros((3, 1))
    expected_cash = None  # not asserted (implementation-dependent)
    expected_shortfall = np.zeros((3, 1))
    expected_agg_shortfall = 0.0
    return L, e0, expected_residual, expected_cash, expected_shortfall, expected_agg_shortfall


def bardoscia_example2():
    # Figure 2: pbar(0)=(10,7,12), e(0)=(12,4,3)
    L = _build_cycle_L(np.array([10.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])

    # Canonical steady state from your scratch:
    expected_residual = np.array([[0.0], [0.0], [12.0]])
    expected_cash = np.array([[2.0], [7.0], [10.0]])  # cash-conserving version
    expected_shortfall = np.array([[0.0], [0.0], [2.0]])
    expected_agg_shortfall = 2.0
    return L, e0, expected_residual, expected_cash, expected_shortfall, expected_agg_shortfall


def bardoscia_example3():
    # Figure 3: pbar(0)=(2,7,12), e(0)=(12,4,3)
    L = _build_cycle_L(np.array([2.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])

    expected_residual = np.array([[0.0], [7.0], [12.0]])
    expected_cash = np.array([[10.0], [6.0], [3.0]])
    expected_shortfall = np.array([[0.0], [1.0], [9.0]])
    expected_agg_shortfall = 10.0
    return L, e0, expected_residual, expected_cash, expected_shortfall, expected_agg_shortfall


"""
import numpy as np

def bardoscia_cycle_V(pbar: np.ndarray) -> np.ndarray:
    Pi = np.array([
        [0.0, 1.0, 0.0],  # A -> B
        [0.0, 0.0, 1.0],  # B -> C
        [1.0, 0.0, 0.0],  # C -> A
    ])
    return np.diag(pbar) @ Pi

def bardoscia_example1():
    V = bardoscia_cycle_V(np.array([10.0, 7.0, 5.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0

def bardoscia_example2():
    V = bardoscia_cycle_V(np.array([10.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0

def bardoscia_example3():
    V = bardoscia_cycle_V(np.array([2.0, 7.0, 12.0]))
    e0 = np.array([12.0, 4.0, 3.0])
    return V, e0
    
"""
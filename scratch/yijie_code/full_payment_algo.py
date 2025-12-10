"""
Full Payment Algorithm

Iterative debt settlement algorithm for financial networks where nodes pay
obligations when they have sufficient cash balances.
"""

import numpy as np


def full_payment_algo(L, e0, TOL_ZERO=1e-8):
    """
    Full payment algorithm for debt settlement.

    Parameters
    ----------
    L : numpy.ndarray
        Liability matrix (N x N). L[i,j] is obligation from node i to node j.
    e0 : numpy.ndarray
        Initial cash balances (N x 1 or N-length array).
    TOL_ZERO : float, optional (default=1e-8)
        Relative tolerance for numerical precision when checking if a node can pay.
        A node can pay if: cash >= obligation * (1 - TOL_ZERO)
        This handles tiny numerical errors from compression solvers (e.g., PuLP/CBC).
        Default of 1e-8 corresponds to ~0.000001% relative error, which is negligible
        for financial applications but sufficient to handle solver precision limits.

    Returns
    -------
    p_bar : numpy.ndarray
        Residual unpaid obligations (N x 1).
    e_steady : numpy.ndarray
        Final cash balances (N x 1).
    p_steady : numpy.ndarray
        Final payments made (N x 1).
    residual_obligation : numpy.ndarray
        Residual obligation matrix (N x N).
    ITER : int
        Number of iterations until convergence.
    p_bar1 : numpy.ndarray
        Residual obligations after first iteration (N x 1).
    """
    # Initializing model
    t = 0
    N = len(e0)

    if e0.ndim == 1:
        e0 = e0.reshape(-1, 1)

    e_list = [e0.copy()]
    p_list = []

    p_hat = L.sum(axis=1).reshape(-1, 1)  # obligations
    e = e0.copy()  # cash balances
    a = np.zeros((N, 1))  # active nodes
    # Allow tolerance: node can pay if cash >= obligation * (1 - TOL_ZERO)
    a[e[:, 0] >= p_hat[:, 0] * (1 - TOL_ZERO)] = 1

    # Relative liability matrix
    pi = np.zeros_like(L, dtype=float)
    for i in range(N):
        if p_hat[i, 0] > 0:
            pi[i, :] = L[i, :] / p_hat[i, 0]

    # First round
    p = a * p_hat
    p_list.append(p.copy())

    incoming = (pi * p[:, 0].reshape(-1, 1)).sum(axis=0).reshape(-1, 1)
    e = e + incoming - p
    e_list.append(e.copy())

    p_hat = (1 - a) * p_hat

    if a.sum() <= 0:
        p_bar1 = p_hat.copy()
    else:
        p_bar1 = None

    # Iterating until convergence
    while a.sum() > 0:
        t += 1

        a = np.zeros((N, 1))
        # Allow tolerance: node can pay if cash >= obligation * (1 - TOL_ZERO)
        a[(e[:, 0] >= p_hat[:, 0] * (1 - TOL_ZERO)) & (p_hat[:, 0] > 0)] = 1

        p = a * p_hat
        p_list.append(p.copy())

        incoming = (pi * p[:, 0].reshape(-1, 1)).sum(axis=0).reshape(-1, 1)
        e = e + incoming - p
        e_list.append(e.copy())

        p_hat = (1 - a) * p_hat

        if t == 1 and p_bar1 is None:
            p_bar1 = p_hat.copy()

    p_bar = p_hat
    e_steady = e_list[-1]
    p_steady = p_list[-1] if p_list else np.zeros((N, 1))
    ITER = len(p_list) - 1
    residual_obligation = pi * p_bar[:, 0].reshape(-1, 1)

    if p_bar1 is None:
        p_bar1 = p_bar.copy()

    return p_bar, e_steady, p_steady, residual_obligation, ITER, p_bar1

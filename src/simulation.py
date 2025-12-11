import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FPAResult:
    """
    Result of running the Full Payment Algorithm (FPA).

    All vectors are N x 1 column arrays.
    """
    residual_obligations: np.ndarray        # l_ss: unpaid obligations at steady state
    final_cash: np.ndarray                  # b_ss: cash balances at steady state
    final_payments: np.ndarray              # p(T): payments made in final iteration
    residual_matrix: np.ndarray             # V_resid = Pi * residual_obligations
    iterations: int                         # number of *updates* after the first round
    first_round_residual: np.ndarray        # l(1): residual obligations after t=0 step    
    shortfall: np.ndarray                   # p_shortfall: firm-level shortfalls max(0, l_ss - b_ss)
    aggregate_shortfall: float              # R = sum_i p_shortfall_i
    # Optional paths
    cash_path: Optional[List[np.ndarray]] = None
    payment_path: Optional[List[np.ndarray]] = None
    indicator_path: Optional[List[np.ndarray]] = None


def run_fpa(
    L: np.ndarray,
    e0: np.ndarray,
    rel_tol: float = 1e-8,
    return_paths: bool = False,
) -> FPAResult:
    """
    Run the Full Payment Algorithm (FPA) as defined in Section 3.3 of the paper,
    with a numerical tolerance on the liquidity condition.  L corresponds to the
    matrix V of realised VM payment obligations; e0 corresponds to b(0). :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    L : np.ndarray
        N x N matrix of payment obligations. L[i, j] is the obligation from node i to node j.
    e0 : np.ndarray
        Initial cash vector b(0). Shape (N,) or (N, 1).
    rel_tol : float, optional
        Relative tolerance for "can pay" condition. Node i is treated as liquid at time t
        if b_i(t) >= l_i(t) * (1 - rel_tol). Default 1e-8.
    return_paths : bool, optional
        If True, also return full time paths of cash, payments and indicators.

    Returns
    -------
    FPAResult
        Dataclass with steady-state outcomes and (optionally) paths.
    """
    # Ensure column-vector shape for e0
    e0 = np.asarray(e0, dtype=float)
    if e0.ndim == 1:
        e0 = e0.reshape(-1, 1)

    L = np.asarray(L, dtype=float)
    N = e0.shape[0]

    # Total obligations l(0)
    l = L.sum(axis=1).reshape(-1, 1)  # l_i = sum_j L[i, j]

    # Relative liability matrix Pi_ij = L_ij / l_i
    Pi = np.zeros_like(L, dtype=float)
    for i in range(N):
        if l[i, 0] > 0.0:
            Pi[i, :] = L[i, :] / l[i, 0]

    # Storage for paths (if requested)
    cash_path: List[np.ndarray] = [e0.copy()] if return_paths else []
    payment_path: List[np.ndarray] = [] if return_paths else []
    indicator_path: List[np.ndarray] = [] if return_paths else []

    # Initial state
    t = 0
    b = e0.copy()                     # b(0)
    I = np.zeros((N, 1))              # I(0)

    # Activation condition at t=0 (first round)
    I[b[:, 0] >= l[:, 0] * (1.0 - rel_tol)] = 1.0

    # Payments p(0) = I(0) * l(0)
    p = I * l

    if return_paths:
        payment_path.append(p.copy())
        indicator_path.append(I.copy())

    # Incoming payments at t=0
    incoming = (Pi * p[:, 0].reshape(-1, 1)).sum(axis=0).reshape(-1, 1)

    # Update cash and liabilities to t=1
    b = b + incoming - p             # b(1)
    l = (1.0 - I) * l                # l(1)

    if return_paths:
        cash_path.append(b.copy())

    # Record residual obligations after the first round
    first_round_residual = l.copy()

    # Iterate until no node can pay in full
    while True:
        t += 1

        # Determine which nodes can pay at time t
        I = np.zeros((N, 1))
        mask_can_pay = (l[:, 0] > 0.0) & (b[:, 0] >= l[:, 0] * (1.0 - rel_tol))
        I[mask_can_pay, 0] = 1.0

        if I.sum() <= 0:
            # No more payments: steady state reached
            break

        # Payments at time t
        p = I * l

        # Incoming payments at time t
        incoming = (Pi * p[:, 0].reshape(-1, 1)).sum(axis=0).reshape(-1, 1)

        # Update cash and liabilities
        b = b + incoming - p
        l = (1.0 - I) * l

        if return_paths:
            payment_path.append(p.copy())
            indicator_path.append(I.copy())
            cash_path.append(b.copy())

    # At this point, l and b are steady-state residual obligations and cash
    residual_obligations = l
    final_cash = b
    final_payments = p if 'p' in locals() else np.zeros((N, 1))
    iterations = max(t - 1, 0)  # number of *updates* after t=0 step

    # Residual obligation matrix: Pi * l_ss
    residual_matrix = Pi * residual_obligations[:, 0].reshape(-1, 1)

    # --- Firm-level and aggregate shortfall (eq. 4–5) ---
    shortfall = np.maximum(0.0, residual_obligations - final_cash)
    aggregate_shortfall = float(shortfall.sum())

    return FPAResult(
        residual_obligations=residual_obligations,
        final_cash=final_cash,
        final_payments=final_payments,
        residual_matrix=residual_matrix,
        iterations=iterations,
        first_round_residual=first_round_residual,
        shortfall=shortfall,
        aggregate_shortfall=aggregate_shortfall,
        cash_path=cash_path if return_paths else None,
        payment_path=payment_path if return_paths else None,
        indicator_path=indicator_path if return_paths else None,
    )

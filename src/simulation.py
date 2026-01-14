# src/simulation.py
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
    activation_round: np.ndarray            # N x 1 ints: 0 (l0=0), 1 (first round), 2+ (later), -1 (never)
    # Optional paths
    cash_path: Optional[List[np.ndarray]] = None
    payment_path: Optional[List[np.ndarray]] = None
    indicator_path: Optional[List[np.ndarray]] = None


def run_fpa(
    L: np.ndarray,
    e0: np.ndarray,
    rel_tol: float = 1e-8,
    return_paths: bool = False,
    validate: bool = True
) -> FPAResult:
    """
    Run the Full Payment Algorithm (FPA) (hard-default clearing with full payment).

    Each node i either pays its *entire* remaining obligation l_i(t) in a payment
    round (if it has sufficient cash), or pays zero. When a node pays, its total
    payment l_i(t) is distributed to creditors proportionally to liabilities via
    the relative liability matrix Pi, where Pi[i, j] = L[i, j] / l_i(0).

    Notation
    --------
    L : (N, N) matrix of obligations, where L[i, j] is the obligation from i to j.
    e0 : (N,) or (N, 1) initial cash vector b(0).
    l_i(t) : remaining total obligations of i at time t.
    b_i(t) : cash of i at time t.

    Payment rule (per round)
    ------------------------
    Node i is deemed able to pay at time t if

        b_i(t) + abs_tol_i(t) >= l_i(t),

    where the numerical tolerance is

        abs_tol_i(t) = rel_tol * max(1, l_i(t)).

    This tolerance is purely to guard against floating-point roundoff when b and l
    are very close; economically, the model is “pay in full or pay nothing”.

    Outputs
    -------
    Returns an FPAResult containing steady-state residual obligations, final cash,
    final payments in the last payment round, residual obligation matrix, firm-level
    shortfalls, aggregate shortfall, and activation_round bookkeeping.

    activation_round coding:
      -1 : never pays
       0 : has zero liabilities at t=0 (l_i(0) = 0)
       1 : pays in the first payment round (t=0 step)
       2+ : first pays in a later round

    Parameters
    ----------
    L : np.ndarray
        (N, N) obligations matrix.
    e0 : np.ndarray
        Initial cash vector, shape (N,) or (N, 1).
    rel_tol : float
        Numerical tolerance factor used to form abs_tol_i(t) = rel_tol * max(1, l_i(t)).
        Default 1e-8.
    return_paths : bool
        If True, also return time paths of cash, payments, and indicators.
    validate : bool
        If True, run internal consistency checks (e.g., nodes that ever pay must end with
        zero residual obligations/shortfall up to tolerance).
    """
    # Ensure column-vector shape for e0
    e0 = np.asarray(e0, dtype=float)
    if e0.ndim == 1:
       e0 = e0.reshape(-1, 1)

    L = np.asarray(L, dtype=float)
    N = e0.shape[0]

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square (N,N); got {L.shape}")
    if e0.shape != (L.shape[0], 1):
        raise ValueError(f"e0 must have shape {(L.shape[0],1)}; got {e0.shape}")

    # Total obligations l(0)
    l = L.sum(axis=1).reshape(-1, 1)  # l_i = sum_j L[i, j]

    # --- Activation round bookkeeping ---
    # -1 = never pays
    #  0 = l(0)=0 (no liabilities)
    #  1 = pays in first round (mask0, no inflows)
    #  2+ = pays in later rounds
    activation_round = np.full((N, 1), -1, dtype=int)

    zero_liab = (l[:, 0] <= 0.0)
    activation_round[zero_liab, 0] = 0

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
    abs_tol0 = rel_tol * np.maximum(1.0, l[:, 0])
    mask0 = (l[:, 0] > 0.0) & (b[:, 0] + abs_tol0 >= l[:, 0]) 
    I[mask0, 0] = 1.0

    # --- Firms that pay in the first round ---
    #mask0_pay = mask0 & (~zero_liab)  # exclude l0=0 (those are coded as 0)
    #activation_round[mask0_pay, 0] = 1
    activation_round[mask0, 0] = 1

    # Payments p(0) = I(0) * l(0)
    p = I * l

    if return_paths:
        payment_path.append(p.copy())
        indicator_path.append(I.copy())

    # Incoming payments at t=0
    incoming = Pi.T @ p


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
        abs_tol = rel_tol * np.maximum(1.0, l[:, 0])
        mask_can_pay = (l[:, 0] > 0.0) & (b[:, 0] + abs_tol >= l[:, 0])

        I[mask_can_pay, 0] = 1.0

        if I.sum() <= 0:
            # No more payments: steady state reached
            break
        
        # --- Record first activation round for newly paying firms ---
        newly = (I[:, 0] > 0.0) & (activation_round[:, 0] == -1)
        if np.any(newly):
            # This loop iteration corresponds to payment round (t+1):
            # t=1 here means "second payment round", so activation round = t+1
            activation_round[newly, 0] = t + 1

        # Payments at time t
        p = I * l

        # Incoming payments at time t
        incoming = Pi.T @ p   

        # Update cash and liabilities     
        b = b + incoming - p
        tiny_neg = (b[:, 0] < 0.0) & (b[:, 0] > -abs_tol)
        b[tiny_neg, 0] = 0.0
        l = (1.0 - I) * l


        if return_paths:
            payment_path.append(p.copy())
            indicator_path.append(I.copy())
            cash_path.append(b.copy())

    # At this point, l and b are steady-state residual obligations and cash
    residual_obligations = l
    final_cash = b
    final_payments = p
    iterations = max(t - 1, 0)  # number of *updates* after t=0 step

    # Residual obligation matrix: Pi * l_ss
    residual_matrix = Pi * residual_obligations[:, 0].reshape(-1, 1)

    # --- Firm-level and aggregate shortfall (eq. 4–5) ---
    shortfall = np.maximum(0.0, residual_obligations - final_cash)
    aggregate_shortfall = float(shortfall.sum())

    
    # --- Internal invariants (debug / validation) ---
    if validate:
        # Nodes that ever paid in any round
        act = activation_round[:, 0]        # shape (N,)
        paid = act >= 1

        # Node-specific tolerance scaled by initial obligation size
        l0 = L.sum(axis=1)                  # shape (N,)
        tol_i = 10.0 * rel_tol * np.maximum(1.0, l0) + 1e-12

        resid = residual_obligations[:, 0]  # shape (N,)
        sf = shortfall[:, 0]                # shape (N,)

        # Invariant 1: anyone who paid must have cleared all obligations
        bad_resid = np.where(paid & (resid > tol_i))[0]
        if bad_resid.size > 0:
            i = bad_resid[np.argmax(resid[bad_resid])]
            raise AssertionError(
                "FPA invariant violated: node that paid has positive residual obligations. "
                f"node={int(i)}, residual={float(resid[i]):.6g}, tol={float(tol_i[i]):.6g}"
            )

        # Invariant 2: anyone who paid must have zero shortfall
        bad_sf = np.where(paid & (sf > tol_i))[0]
        if bad_sf.size > 0:
            i = bad_sf[np.argmax(sf[bad_sf])]
            raise AssertionError(
                "FPA invariant violated: node that paid has positive shortfall. "
                f"node={int(i)}, shortfall={float(sf[i]):.6g}, tol={float(tol_i[i]):.6g}"
            )

    
    return FPAResult(
        residual_obligations=residual_obligations,
        final_cash=final_cash,
        final_payments=final_payments,
        residual_matrix=residual_matrix,
        iterations=iterations,
        first_round_residual=first_round_residual,
        shortfall=shortfall,
        aggregate_shortfall=aggregate_shortfall,
        activation_round=activation_round,  
        cash_path=cash_path if return_paths else None,
        payment_path=payment_path if return_paths else None,
        indicator_path=indicator_path if return_paths else None,
    )

import numpy as np


def behavioural_base_from_V(V: np.ndarray) -> np.ndarray:
    """
    Behavioural buffer *shape* based on outgoing VM obligations.

    In the reduced-form setup where we treat V as the realised VM liabilities matrix,
    a natural proxy for "gross notional scale" is outgoing VM payable:

        p_i = sum_j V[i, j]

    This returns the base shape vector x(V) = p, which is then scaled by a
    global factor alpha:

        b(alpha) = alpha * x(V)

    Returns
    -------
    base : np.ndarray
        Shape (N,) vector of non-negative base buffer weights.
    """
    V = np.asarray(V, dtype=float)
    base = V.sum(axis=1)
    return base

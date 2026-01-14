import numpy as np
from typing import Iterable
from typing import Literal

__all__ = ["lorenz_points", "gini", "topk_shares"]


def lorenz_points(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Lorenz curve points for nonnegative values.

    Returns x,y arrays including (0,0) and (1,1).
    x = cumulative share of nodes
    y = cumulative share of total value
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 0.0)

    n = v.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    total = float(v.sum())
    if total <= 0:
        x = np.linspace(0.0, 1.0, n + 1)
        y = np.linspace(0.0, 1.0, n + 1)
        return x, y

    v_sorted = np.sort(v)  # ascending
    cum = np.cumsum(v_sorted)

    x = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    y = np.concatenate(([0.0], cum / total))
    return x, y


def gini(x: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Gini coefficient for a nonnegative 1D array.
    Returns 0 if all mass is zero.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= eps:
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    # G = (2 * sum_i i*x_i)/(n*sum x) - (n+1)/n  with i=1..n
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (i * x_sorted).sum()) / (n * s) - (n + 1.0) / n)


def topk_shares(values: np.ndarray, ks: Iterable[int] = (1, 5, 10, 20)) -> dict[str, float]:
    """
    Compute top-k shares of total from a nonnegative vector.
    Returns dict with 'total' and 'top{k}_share' entries.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 0.0)

    total = float(v.sum())
    out: dict[str, float] = {"total": total}
    if total <= 0 or v.size == 0:
        for k in ks:
            out[f"top{k}_share"] = float("nan")
        return out

    v_desc = np.sort(v)[::-1]
    for k in ks:
        kk = min(int(k), v_desc.size)
        out[f"top{k}_share"] = float(v_desc[:kk].sum() / total)
    return out


HHIMode = Literal["incoming", "outgoing"]

def flow_hhi(
    V: np.ndarray,
    *,
    mode: HHIMode = "incoming",
) -> np.ndarray:
    """
    Compute node-level Herfindahl–Hirschman Index (HHI) for flows.

    Parameters
    ----------
    V : (N, N) ndarray
        Flow matrix where V[i, j] is payment from i to j.
    mode : {"incoming", "outgoing"}
        - "incoming": concentration of incoming payments (column-wise)
        - "outgoing": concentration of outgoing payments (row-wise)

    Returns
    -------
    hhi : (N,) ndarray
        HHI for each node. Zero for nodes with zero total flow in the chosen mode.
    """
    V = np.asarray(V, dtype=float)

    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"V must be square (N,N); got shape {V.shape}")

    if mode == "incoming":
        total = V.sum(axis=0)  # (N,)
        denom = total[None, :]  # (1,N) broadcast to (N,N)
        shares = np.zeros_like(V, dtype=float)
        np.divide(V, denom, out=shares, where=total[None, :] > 0)
        return (shares ** 2).sum(axis=0)

    if mode == "outgoing":
        total = V.sum(axis=1)  # (N,)
        denom = total[:, None]  # (N,1) broadcast to (N,N)
        shares = np.zeros_like(V, dtype=float)
        np.divide(V, denom, out=shares, where=total[:, None] > 0)
        return (shares ** 2).sum(axis=1)

    raise ValueError(f"Unknown mode '{mode}'. Use 'incoming' or 'outgoing'.")
"""
Compression routines for PaymentNetwork objects.

This module wraps the CDFD decomposition methods so that we can apply
BFF-based and max-C (maximal compression) to our PaymentNetwork
representation.
"""

from dataclasses import dataclass
from typing import Any

import warnings
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components

from .networks import PaymentNetwork
from . import cdfd  # currently using local CDFD implementation/toolbox


@dataclass
class CompressionResult:
    """
    Result of applying a compression method to a PaymentNetwork.
    """
    compressed: PaymentNetwork          # compressed network (directional part)
    method: str                         # e.g. "BFF", "maxC_pulp", "maxC_ortools"
    gross_before: float                 # Σ_ij W_ij before
    gross_after: float                  # Σ_ij W_ij after
    savings_abs: float                  # gross_before - gross_after
    savings_frac: float                 # savings_abs / gross_before (if > 0)
    C_circular: csr_array | None        # circular component C (optional)
    D_directional: csr_array            # directional component D
    meta: dict[str, Any]                # extra diagnostics (iterations, etc.)


def _matrix_to_payment_network(
    G: PaymentNetwork,
    D: csr_array,
) -> PaymentNetwork:
    """
    Build a new PaymentNetwork with the same node sets but new weights D.
    """
    W_new = D.toarray() if not isinstance(D, np.ndarray) else D
    return PaymentNetwork(
        W=W_new,
        source_nodes=list(G.source_nodes),
        core_nodes=list(G.core_nodes),
        sink_nodes=list(G.sink_nodes),
    )


def _build_result(
    G: PaymentNetwork,
    method: str,
    C: csr_array | None,
    D: csr_array,
    meta: dict[str, Any] | None = None,
) -> CompressionResult:
    gross_before = float(np.sum(G.W))
    gross_after = float(np.sum(D))
    savings_abs = gross_before - gross_after

    # Clean up tiny floating-point noise
    if abs(savings_abs) < 1e-10:
        savings_abs = 0.0

    savings_frac = savings_abs / gross_before if gross_before > 0 else 0.0

    compressed_net = _matrix_to_payment_network(G, D)

    return CompressionResult(
        compressed=compressed_net,
        method=method,
        gross_before=gross_before,
        gross_after=gross_after,
        savings_abs=savings_abs,
        savings_frac=savings_frac,
        C_circular=C,
        D_directional=D,
        meta=meta or {},
    )


def compress_BFF(
    G: PaymentNetwork,
    tol_zero: float = 1e-12,
    require_conservative: bool = True,
    require_full_conservative: bool = True,
) -> CompressionResult:
    """
    Apply BFF-based CDFD to obtain a compression of G.

    This uses the CDFD_BFF method which iteratively removes circulations
    via BFF until the residual directional component is acyclic. Compression is
    achieved by discarding the circular part C and retaining D.

    Parameters
    ----------
    G : PaymentNetwork
        Original network.
    tol_zero : float
        Tolerance for treating small weights as zero inside the CDFD_BFF solver.

    Returns
    -------
    CompressionResult
        Contains the compressed PaymentNetwork and summary statistics.
    """
    W = csr_array(G.W)
    C, D = cdfd.CDFD_BFF(W, TOL_ZERO=tol_zero)

    res = _build_result(G, method="BFF", C=C, D=D, meta={})

    if require_full_conservative and not require_conservative:
        warnings.warn(
            "require_full_conservative=True supersedes require_conservative=False; "
            "full conservative validation includes conservative constraints.",
            UserWarning,
        )

    if require_full_conservative:
        # full conservative implies conservative + acyclic
        validate_full_conservative(G, res)
    elif require_conservative:
        validate_conservative_compression(G, res)

    return res


def compress_maxC(
    G: PaymentNetwork,
    tol_zero: float = 1e-12,
    solver: str = "ortools",
    require_conservative: bool = True,
    require_full_conservative: bool = True,
) -> CompressionResult:
    """
    Apply max-C (maximal compression) using min-cost flow.

    - solver="ortools" (default): uses integer-based min-cost flow via OR-Tools,
      suitable for money-like data (scaled to a fixed number of decimals).
    - solver="pulp": uses PuLP + CBC. This may not work on Apple Silicon
      unless a compatible CBC binary is installed and configured.
    """
    W = csr_array(G.W)

    if solver == "pulp":
        C, D = _maxC_pulp(W, tol_zero=tol_zero)
        method_name = "maxC_pulp"
    elif solver == "ortools":
        C, D = _maxC_ortools(W, tol_zero=tol_zero)
        method_name = "maxC_ortools"
    else:
        raise NotImplementedError(f"solver='{solver}' is not yet implemented.")

    res = _build_result(G, method=method_name, C=C, D=D, meta={"solver": solver})

    if require_full_conservative and not require_conservative:
        warnings.warn(
            "require_full_conservative=True supersedes require_conservative=False; "
            "full conservative validation includes conservative constraints.",
            UserWarning,
        )

    if require_full_conservative:
        # full conservative implies conservative + acyclic
        validate_full_conservative(G, res)
    elif require_conservative:
        validate_conservative_compression(G, res)

    return res


def _maxC_ortools(W: csr_array, tol_zero: float) -> tuple[csr_array, csr_array]:
    """
    Thin wrapper around cdfd.CDFD_min_cost_ortools to match our internal interface.

    The underlying implementation scales weights to integer-like values,
    runs min-cost flow via OR-Tools, then rescales back. It is designed
    for money-like data with a fixed number of decimal places.
    """
    C, D = cdfd.CDFD_min_cost_ortools(W, TOL_ZERO=tol_zero)
    return C, D


def _maxC_pulp(W: csr_array, tol_zero: float) -> tuple[csr_array, csr_array]:
    """
    Thin wrapper around cdfd.CDFD_min_cost_pulp to match our internal interface.
    """
    C, D = cdfd.CDFD_min_cost_pulp(W, TOL_ZERO=tol_zero)
    return C, D


def validate_conservative_compression(
    G_before: PaymentNetwork,
    result: CompressionResult,
    tol: float = 1e-10,
) -> None:
    """
    Validate that a compression result satisfies conservative compression constraints:

    1. Net positions are preserved (within tol).
    2. No new counterparties are created: support(W_after) ⊆ support(W_before).
    3. Weights remain non-negative (within tol).
    4. Gross notional does not increase (within tol).

    Raises
    ------
    ValueError
        If any of the constraints is violated.
    """
    W0 = np.asarray(G_before.W)
    W1 = np.asarray(result.compressed.W)

    # 1. Net positions preserved
    net0 = G_before.net_positions
    net1 = result.compressed.net_positions
    net_diff = np.max(np.abs(net0 - net1))
    if net_diff > tol:
        raise ValueError(
            f"Compression '{result.method}' violates net-position preservation: "
            f"max |Δnet| = {net_diff}"
        )

    # 2. Conservative: no new counterparties
    # "edge exists" = weight > tol
    new_edges_mask = (W1 > tol) & (W0 <= tol)
    if np.any(new_edges_mask):
        i_new, j_new = np.where(new_edges_mask)
        raise ValueError(
            f"Compression '{result.method}' creates {len(i_new)} new edges "
            f"that were not present in the original network."
        )

    # 3. Non-negative weights
    min_weight_after = np.min(W1)
    if min_weight_after < -tol:
        raise ValueError(
            f"Compression '{result.method}' produced negative weights: "
            f"min W_after = {min_weight_after}"
        )

    # 4. Gross notional does not increase
    if result.gross_after - result.gross_before > tol:
        raise ValueError(
            f"Compression '{result.method}' increases gross notional: "
            f"before={result.gross_before}, after={result.gross_after}"
        )


def _is_acyclic_matrix(W: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check if a weighted adjacency matrix represents a DAG (no directed cycles).

    We treat entries with |w_ij| <= tol as zero.
    A directed graph is acyclic iff every strongly connected component
    has size 1 and there are no self-loops.
    """
    # Convert to sparse and clean tiny entries
    W_sp = csr_array(W)
    data = W_sp.data
    data[np.abs(data) <= tol] = 0.0
    W_sp.data = data
    W_sp.eliminate_zeros()

    # Self-loops => definitely cyclic
    if np.any(np.abs(W_sp.diagonal()) > tol):
        return False

    n_nodes = W_sp.shape[0]
    n_components, labels = connected_components(
        csgraph=W_sp,
        directed=True,
        connection="strong",
    )

    # A DAG has no strongly connected component with size > 1
    # i.e. number of SCCs equals number of nodes.
    return n_components == n_nodes


def validate_full_conservative(
    G_before: PaymentNetwork,
    result: CompressionResult,
    tol: float = 1e-10,
) -> None:
    """
    Validate that a compression is both conservative AND fully compressing
    (i.e. the resulting payment network is acyclic).

    Conservative constraints:
      - net positions preserved
      - no new counterparties
      - non-negative weights
      - gross notional does not increase

    Full conservative compression (here):
      - compressed network has no directed cycles (is a DAG)

    Raises
    ------
    ValueError
        If any of the constraints is violated.
    """
    # First check conservative constraints
    validate_conservative_compression(G_before, result, tol=tol)

    # Then check acyclicity
    W1 = np.asarray(result.compressed.W)
    if not _is_acyclic_matrix(W1, tol=tol):
        raise ValueError(
            f"Compression '{result.method}' is not acyclic: "
            "compressed network still contains directed cycles."
        )



def compress(
    G: PaymentNetwork,
    *,
    method: str = "bff",
    tol_zero: float = 1e-12,
    # maxC options
    solver: str = "ortools",
    # validation options
    require_conservative: bool = True,
    require_full_conservative: bool = True,
) -> CompressionResult:
    """
    Unified compression wrapper.

    Parameters
    ----------
    G : PaymentNetwork
        Network to compress.
    method : str
        Compression method:
          - "bff"  : BFF-based CDFD compression
          - "maxc" : maximal compression (min-cost flow)
    tol_zero : float
        Numerical tolerance passed to the underlying solver.
    solver : str
        Only used for method="maxc". One of {"ortools","pulp"}.
    require_conservative, require_full_conservative : bool
        Validation flags.

    Returns
    -------
    CompressionResult
    """
    m = method.strip().lower()

    if m in {"bff", "cdfd_bff"}:
        return compress_BFF(
            G,
            tol_zero=tol_zero,
            require_conservative=require_conservative,
            require_full_conservative=require_full_conservative,
        )

    if m in {"maxc", "max_c", "max-c"}:
        return compress_maxC(
            G,
            tol_zero=tol_zero,
            solver=solver,
            require_conservative=require_conservative,
            require_full_conservative=require_full_conservative,
        )

    raise ValueError(
        f"Unknown compression method '{method}'. "
        "Use one of: 'bff', 'maxc'."
    )

"""
Network construction utilities for the compression–liquidity model.

Implements a three-tier OTC-style network with:
- source nodes (only outgoing exposures),
- core nodes (inter-dealer network),
- sink nodes (only incoming exposures),
and provides functions to attach weights to a binary structure.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class PaymentNetwork:
    """
    Minimal representation of a VM obligation network.

    Attributes
    ----------
    W : np.ndarray
        Weighted adjacency matrix where W[i, j] is the obligation of i to j.
    source_nodes : list[int]
        Indices of source nodes (only outgoing edges).
    core_nodes : list[int]
        Indices of core nodes (dealers).
    sink_nodes : list[int]
        Indices of sink nodes (only incoming edges).
    """
    W: np.ndarray
    source_nodes: List[int]
    core_nodes: List[int]
    sink_nodes: List[int]

    @property
    def num_nodes(self) -> int:
        return self.W.shape[0]

    @property
    def gross_notional(self) -> float:
        return float(self.W.sum())

    @property
    def net_positions(self) -> np.ndarray:
        # outflows - inflows
        return self.W.sum(axis=1) - self.W.sum(axis=0)


def generate_three_tier_structure(
    n_core: int,
    n_source: int,
    n_sink: int,
    p: float,
    rng: np.random.Generator | None = None,
    degree_mode: str = "bernoulli",
) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    """
    Generate a three-tier network with binary adjacency (0/1) structure.

    Parameters
    ----------
    n_core : int
        Number of core nodes (dealers).
    n_source : int
        Number of source nodes (ultimate sellers / liquidity providers).
    n_sink : int
        Number of sink nodes (ultimate buyers / end users).
    p : float
        Connection-density parameter.
    rng : np.random.Generator, optional
        Random number generator used for topology draws (edge presence / target choice).
        If None, a fresh generator is created (non-reproducible unless seeded externally).
    degree_mode : {"bernoulli", "fixed"}, optional
        - "bernoulli": each potential edge is present independently with prob p,
          matching the modelling framework of D'Errico & Roukny (2017).
        - "fixed": each source/sink connects to approximately p * n_core distinct
          core nodes, reproducing the behaviour of the earlier implementation.

    Returns
    -------
    L : np.ndarray
        Binary adjacency matrix (0/1) indicating presence of edges.
    source_nodes, core_nodes, sink_nodes : lists of int
        Index sets for the three tiers.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_nodes = n_source + n_core + n_sink

    source_nodes = list(range(n_source))
    core_nodes = list(range(n_source, n_source + n_core))
    sink_nodes = list(range(n_source + n_core, total_nodes))

    L = np.zeros((total_nodes, total_nodes), dtype=float)

    # --- Source → core ---
    if degree_mode == "bernoulli":
        for s in source_nodes:
            for c in core_nodes:
                if rng.random() < p:
                    L[s, c] = 1.0
    elif degree_mode == "fixed":
        num_connections = max(1, int(round(n_core * p)))
        for s in source_nodes:
            if num_connections >= len(core_nodes):
                targets = core_nodes
            else:
                targets = rng.choice(core_nodes, size=num_connections, replace=False)
            L[s, targets] = 1.0
    else:
        raise ValueError(f"Unknown degree_mode={degree_mode!r}")

    # --- Core ↔ core (always Bernoulli ER) ---
    for i in core_nodes:
        for j in core_nodes:
            if i != j and rng.random() < p:
                L[i, j] = 1.0

    # --- Core → sink ---
    if degree_mode == "bernoulli":
        for c in core_nodes:
            for t in sink_nodes:
                if rng.random() < p:
                    L[c, t] = 1.0
    elif degree_mode == "fixed":
        num_connections = max(1, int(round(n_core * p)))
        for t in sink_nodes:
            if num_connections >= len(core_nodes):
                sources_to_sink = core_nodes
            else:
                sources_to_sink = rng.choice(core_nodes, size=num_connections, replace=False)
            L[sources_to_sink, t] = 1.0

    return L, source_nodes, core_nodes, sink_nodes


def assign_weights(
    L: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    weight_mode: str = "pareto",
    alpha: float = 2.0,
    scale: float = 1.0,
    w_min: float = 0.5,
    w_max: float = 1.5,
    constant_weight: float = 1.0,
    round_to: float | None = None,
) -> np.ndarray:
    """
    Assign positive weights to the edges of a binary adjacency matrix.

    Parameters
    ----------
    L : np.ndarray
        Binary adjacency matrix (0/1) indicating presence of edges.
    rng : np.random.Generator, optional
        Random number generator used for weight draws on existing edges.
        If None, a fresh generator is created.
    weight_mode : {"pareto", "uniform", "constant"}, optional
        Choice of weight distribution:
        - "pareto"   : heavy-tailed Pareto-type distribution (default).
        - "uniform"  : Uniform[w_min, w_max].
        - "constant" : all edges equal to `constant_weight`.
    alpha : float, optional
        Tail exponent for the Pareto-like distribution (default: 2.0).
    scale : float, optional
        Global scale factor for the Pareto weights.
    w_min, w_max : float, optional
        Bounds for the uniform distribution when weight_mode="uniform".
    constant_weight : float, optional
        Edge weight when weight_mode="constant".
    round_to : float or None, optional
        If not None, round weights to nearest multiple of `round_to`
        (e.g. 0.01 for cents or 0.001 if units are millions).

    Returns
    -------
    W : np.ndarray
        Weighted adjacency matrix with W[i, j] > 0 where L[i, j] = 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    W = L.copy()
    edges = np.where(L > 0)

    for i, j in zip(edges[0], edges[1]):
        if weight_mode == "pareto":
            u = rng.uniform(0.001, 1.0)
            weight = scale * (1.0 / u) ** (1.0 / alpha)
            weight = min(weight, 1000 * scale)
        elif weight_mode == "uniform":
            weight = rng.uniform(w_min, w_max)
        elif weight_mode == "constant":
            weight = constant_weight
        else:
            raise ValueError(f"Unknown weight_mode={weight_mode!r}")

        if round_to is not None:
            weight = round_to * np.round(weight / round_to)

        W[i, j] = weight

    return W


def generate_three_tier_network(
    n_core: int,
    n_source: int,
    n_sink: int,
    p: float,
    *,
    weight_mode: str = "pareto",
    alpha_weights: float = 2.0,
    scale_weights: float = 1.0,
    rng_topology: np.random.Generator | None = None,
    rng_weights: np.random.Generator | None = None,
    degree_mode: str = "bernoulli",
    round_to: float | None = None,
) -> PaymentNetwork:
    """
      Generate a full three-tier weighted payment network.

    This wrapper enforces independence between:
      (i) topology randomness (which edges exist), and
      (ii) weight randomness (edge weights conditional on topology),

    by accepting two separate RNG streams: `rng_topology` and `rng_weights`.

    Parameters
    ----------
    n_core, n_source, n_sink : int
        Numbers of core, source, and sink nodes.
    p : float
        Connection-density parameter used in the topology generator.
    weight_mode : {"pareto", "uniform", "constant"}, optional
        Distribution used for edge weights. Default "pareto".
    alpha_weights, scale_weights : float, optional
        Parameters for the Pareto-type weights when weight_mode="pareto".
    rng_topology : np.random.Generator, optional
        RNG used only for topology draws (adjacency / degree realisations).
        If None, a fresh generator is created.
    rng_weights : np.random.Generator, optional
        RNG used only for weight draws on realised edges.
        If None, a fresh generator is created.
    degree_mode : {"bernoulli", "fixed"}, optional
        Controls source/core/sink degrees:
        - "bernoulli": each edge present independently with prob p (default).
        - "fixed": each source/sink connects to ~p * n_core cores.
    round_to : float or None, optional
        If not None, weights are rounded to the nearest multiple of `round_to`.

    Returns
    -------
    PaymentNetwork
        Object containing the weighted adjacency matrix and tier index sets.
    
    """
    if rng_topology is None:
        rng_topology = np.random.default_rng()
    if rng_weights is None:
        rng_weights = np.random.default_rng()

    L, source_nodes, core_nodes, sink_nodes = generate_three_tier_structure(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng_topology,
        degree_mode=degree_mode,
    )
    W = assign_weights(
        L,
        rng=rng_weights,
        weight_mode=weight_mode,
        alpha=alpha_weights,
        scale=scale_weights,
        round_to=round_to,
    )
    return PaymentNetwork(
        W=W,
        source_nodes=source_nodes,
        core_nodes=core_nodes,
        sink_nodes=sink_nodes,
    )



def extract_largest_component(G: PaymentNetwork) -> PaymentNetwork:
    """
    Restrict a PaymentNetwork to its largest weakly connected component.

    Treats the directed network as undirected for connectivity purposes:
    an undirected edge exists between i and j if either W[i, j] > 0 or W[j, i] > 0.

    Nodes that are not in the largest component (including isolated nodes)
    are dropped. Tier index sets (source/core/sink) are remapped accordingly.

    If the network is already connected, the original object is returned unchanged.

    Parameters
    ----------
    G : PaymentNetwork
        Original network.

    Returns
    -------
    PaymentNetwork
        Network induced by the largest weakly connected component.
    """
    W = G.W
    n = W.shape[0]
    if n == 0:
        return G

    # Build undirected adjacency for weak connectivity
    A = (W > 0) | (W.T > 0)
    A = A.astype(bool)

    visited = np.zeros(n, dtype=bool)
    components = []

    for start in range(n):
        if not visited[start]:
            stack = [start]
            visited[start] = True
            comp = [start]

            while stack:
                i = stack.pop()
                neighbors = np.nonzero(A[i])[0]
                for j in neighbors:
                    if not visited[j]:
                        visited[j] = True
                        stack.append(j)
                        comp.append(j)

            components.append(comp)

    if not components:
        return G

    # Pick largest component (break ties by first encountered)
    largest = max(components, key=len)

    # If already fully connected, return as-is
    if len(largest) == n:
        return G

    # Sort indices for deterministic ordering
    largest_sorted = sorted(largest)

    # Build index mapping old -> new
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(largest_sorted)}

    # Induce submatrix
    W_sub = W[np.ix_(largest_sorted, largest_sorted)]

    def remap(nodes: list[int]) -> list[int]:
        return [index_map[i] for i in nodes if i in index_map]

    new_source_nodes = remap(G.source_nodes)
    new_core_nodes = remap(G.core_nodes)
    new_sink_nodes = remap(G.sink_nodes)

    return PaymentNetwork(
        W=W_sub,
        source_nodes=new_source_nodes,
        core_nodes=new_core_nodes,
        sink_nodes=new_sink_nodes,
    )

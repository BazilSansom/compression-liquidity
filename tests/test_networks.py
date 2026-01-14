from __future__ import annotations

import numpy as np

from src.networks import generate_three_tier_network, extract_largest_component


def test_three_tier_structure_source_sink_constraints() -> None:
    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)

    G = generate_three_tier_network(
        n_core=5,
        n_source=3,
        n_sink=4,
        p=0.5,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    # Sources: no incoming edges (no one owes them, by construction)
    assert np.all(G.W[:, G.source_nodes] == 0.0)

    # Sinks: no outgoing edges (they owe no one, by construction)
    assert np.all(G.W[G.sink_nodes, :] == 0.0)

    # Net positions sum to ~0 in any closed liabilities matrix
    assert abs(float(G.net_positions.sum())) < 1e-8


def test_degree_mode_fixed_vs_bernoulli_runs() -> None:
    # Mostly a "doesn't error" + "isn't degenerate" test.
    # Use the same RNG seeds for topology so the comparison is stable.
    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)

    G_bern = generate_three_tier_network(
        n_core=5,
        n_source=3,
        n_sink=4,
        p=0.5,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)
    G_fixed = generate_three_tier_network(
        n_core=5,
        n_source=3,
        n_sink=4,
        p=0.5,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="fixed",
        weight_mode="pareto",
    )

    assert float(G_bern.W.sum()) > 0.0
    assert float(G_fixed.W.sum()) > 0.0


def test_weight_modes_basic_properties() -> None:
    n_core, n_source, n_sink, p = 5, 3, 4, 0.5

    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)
    G_pareto = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="pareto",
        alpha_weights=2.0,
        scale_weights=1.0,
    )
    w_pareto = G_pareto.W[G_pareto.W > 0]
    assert w_pareto.size > 0
    assert np.all(w_pareto > 0.0)

    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)
    G_uniform = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="uniform",
    )
    w_uniform = G_uniform.W[G_uniform.W > 0]
    assert w_uniform.size > 0
    assert np.all(w_uniform > 0.0)
    assert float(np.std(w_uniform)) >= 0.0  # sanity (non-negative)

    rng_topology = np.random.default_rng(123)
    rng_weights = np.random.default_rng(456)
    G_const = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="constant",
    )
    w_const = G_const.W[G_const.W > 0]
    assert w_const.size > 0
    assert np.allclose(w_const, w_const[0])  # all equal


def test_extract_largest_component_removes_isolates() -> None:
    rng_topology = np.random.default_rng(42)
    rng_weights = np.random.default_rng(4242)
    G = generate_three_tier_network(
        n_core=20,
        n_source=20,
        n_sink=20,
        p=0.05,
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    G_lcc = extract_largest_component(G)

    # In the returned component, every node should have at least one incident edge
    adj = (G_lcc.W > 0).astype(int)
    deg_total = adj.sum(axis=1) + adj.sum(axis=0)
    assert np.all(deg_total > 0), "Largest component should not contain isolated nodes."

    # And size should be <= original
    assert G_lcc.num_nodes <= G.num_nodes
    assert G_lcc.W.shape[0] == G_lcc.W.shape[1]


def test_topology_invariant_to_weight_rng() -> None:
    """
    With the same topology RNG, the binary edge structure should be identical
    even if we change the weight RNG. This is the key design guarantee of the
    rng_topology / rng_weights split.
    """
    rng_topology_1 = np.random.default_rng(123)
    rng_weights_1 = np.random.default_rng(111)
    G1 = generate_three_tier_network(
        n_core=10,
        n_source=5,
        n_sink=5,
        p=0.4,
        rng_topology=rng_topology_1,
        rng_weights=rng_weights_1,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    rng_topology_2 = np.random.default_rng(123)  # SAME topology seed
    rng_weights_2 = np.random.default_rng(222)   # DIFFERENT weights seed
    G2 = generate_three_tier_network(
        n_core=10,
        n_source=5,
        n_sink=5,
        p=0.4,
        rng_topology=rng_topology_2,
        rng_weights=rng_weights_2,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    # Compare binary adjacency (edge existence)
    A1 = (G1.W > 0).astype(int)
    A2 = (G2.W > 0).astype(int)
    assert np.array_equal(A1, A2), "Topology should not depend on rng_weights."

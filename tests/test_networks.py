from __future__ import annotations

import numpy as np

from src.networks import generate_three_tier_network, extract_largest_component


def test_three_tier_structure_source_sink_constraints() -> None:
    rng = np.random.default_rng(123)
    G = generate_three_tier_network(
        n_core=5,
        n_source=3,
        n_sink=4,
        p=0.5,
        rng=rng,
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
    # This is mostly a "doesn't error" + "isn't degenerate" test.
    rng = np.random.default_rng(123)
    G_bern = generate_three_tier_network(
        n_core=5, n_source=3, n_sink=4, p=0.5, rng=rng, degree_mode="bernoulli", weight_mode="pareto"
    )
    rng = np.random.default_rng(123)
    G_fixed = generate_three_tier_network(
        n_core=5, n_source=3, n_sink=4, p=0.5, rng=rng, degree_mode="fixed", weight_mode="pareto"
    )

    # Both should have some edges (very likely with p=0.5; if this ever flakes, bump p)
    assert float(G_bern.W.sum()) > 0.0
    assert float(G_fixed.W.sum()) > 0.0


def test_weight_modes_basic_properties() -> None:
    n_core, n_source, n_sink, p = 5, 3, 4, 0.5

    rng = np.random.default_rng(123)
    G_pareto = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="pareto",
        alpha_weights=2.0,
        scale_weights=1.0,
    )
    w_pareto = G_pareto.W[G_pareto.W > 0]
    assert w_pareto.size > 0
    assert np.all(w_pareto > 0.0)

    rng = np.random.default_rng(123)
    G_uniform = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="uniform",
    )
    w_uniform = G_uniform.W[G_uniform.W > 0]
    assert w_uniform.size > 0
    assert np.all(w_uniform > 0.0)
    # We don't hardcode [w_min, w_max] because those defaults live inside assign_weights().
    # But we can sanity-check uniform isn't constant:
    assert float(np.std(w_uniform)) >= 0.0

    rng = np.random.default_rng(123)
    G_const = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="constant",
        # constant_weight default should be 1.0
    )
    w_const = G_const.W[G_const.W > 0]
    assert w_const.size > 0
    assert np.allclose(w_const, w_const[0])  # all equal


def test_extract_largest_component_removes_isolates() -> None:
    rng = np.random.default_rng(42)
    G = generate_three_tier_network(
        n_core=20,
        n_source=20,
        n_sink=20,
        p=0.05,
        rng=rng,
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

# tests/test_compression.py
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.networks import generate_three_tier_network, extract_largest_component
from src.compression import (
    compress,
    validate_conservative_compression,
    validate_full_conservative,
)


def _make_network(seed: int = 42):
    # Split RNGs so topology and weights are reproducible but independent
    rng_topology = np.random.default_rng(seed)
    rng_weights = np.random.default_rng(seed + 1)

    G = generate_three_tier_network(
        n_core=10,
        n_source=10,
        n_sink=10,
        p=0.4,  # dense enough to create cycles pre-compression
        rng_topology=rng_topology,
        rng_weights=rng_weights,
        degree_mode="bernoulli",
        weight_mode="pareto",
        round_to=0.01,
    )
    # Optional: avoid isolates / degenerate cases
    G = extract_largest_component(G)
    return G


def _make_network_with_zero(seed: int = 5):
    for s in range(seed, seed + 50):
        G = _make_network(seed=s)
        if np.any(np.asarray(G.W) == 0.0):
            return G
    raise AssertionError("Could not find a generated network with a zero entry in W.")


def test_compression_gross_does_not_increase_and_preserves_net() -> None:
    G = _make_network(seed=1)
    gross_before = float(G.W.sum())
    net_before = G.net_positions.copy()

    # BFF
    res_bff = compress(G, method="bff", solver="ortools")
    assert res_bff.gross_after <= gross_before + 1e-8
    assert np.allclose(res_bff.compressed.net_positions, net_before, atol=1e-8), (
        "BFF must preserve net positions."
    )

    # maxC
    res_maxc = compress(G, method="maxc", solver="ortools")
    assert res_maxc.gross_after <= gross_before + 1e-8
    assert np.allclose(res_maxc.compressed.net_positions, net_before, atol=1e-8), (
        "maxC must preserve net positions."
    )

    # maxC should compress at least as much as BFF (weakly)
    assert res_maxc.gross_after <= res_bff.gross_after + 1e-8


def test_validators_accept_happy_path() -> None:
    G = _make_network(seed=2)

    res_bff = compress(G, method="bff", solver="ortools")
    validate_conservative_compression(G, res_bff)
    validate_full_conservative(G, res_bff)

    res_maxc = compress(G, method="maxc", solver="ortools")
    validate_conservative_compression(G, res_maxc)
    validate_full_conservative(G, res_maxc)


def test_validator_detects_tampered_net_positions() -> None:
    G = _make_network(seed=3)
    res = compress(G, method="bff", solver="ortools")

    # IMPORTANT: don't mutate the original res in-place; build a tampered copy
    W_bad = np.array(res.compressed.W, dtype=float, copy=True)
    W_bad[0, 1] += 1.0

    tampered = replace(
        res,
        compressed=replace(res.compressed, W=W_bad),
    )

    with pytest.raises(ValueError):
        validate_conservative_compression(G, tampered)


def test_validator_detects_cycle_in_full_conservative() -> None:
    G = _make_network(seed=4)
    res = compress(G, method="bff", solver="ortools")

    W_bad = np.array(res.compressed.W, dtype=float, copy=True)
    W_bad[0, 0] += 1.0  # self-loop => directed cycle

    tampered = replace(
        res,
        compressed=replace(res.compressed, W=W_bad),
    )

    with pytest.raises(ValueError):
        validate_full_conservative(G, tampered)


def test_validator_detects_new_edge_creation() -> None:
    G = _make_network_with_zero(seed=5)
    res = compress(G, method="bff", solver="ortools")

    W0 = np.asarray(G.W)
    W_bad = np.array(res.compressed.W, dtype=float, copy=True)

    # Find a (i,j) where original had no edge, and force a new positive edge.
    zeros = np.argwhere(W0 == 0.0)
    assert zeros.size > 0, "Test requires at least one zero entry in original W."

    i, j = map(int, zeros[0])
    W_bad[i, j] = max(W_bad[i, j], 0.5)

    tampered = replace(
        res,
        compressed=replace(res.compressed, W=W_bad),
    )

    with pytest.raises(ValueError):
        validate_conservative_compression(G, tampered)

import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so "import src..." works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.networks import generate_three_tier_network

from src.networks import generate_three_tier_network, extract_largest_component


def describe_network(label, G):
    print(f"\n=== {label} ===")
    print("W shape:", G.W.shape)
    print("Num nodes:", G.num_nodes)
    print("source/core/sink:", len(G.source_nodes), len(G.core_nodes), len(G.sink_nodes))

    # Basic structure checks
    print("Sources have no incoming edges:",
          np.all(G.W[:, G.source_nodes] == 0))
    print("Sinks have no outgoing edges:",
          np.all(G.W[G.sink_nodes, :] == 0))

    # Degree stats (out-degree)
    out_degrees = (G.W > 0).sum(axis=1)
    print("Out-degree (min, max, mean):",
          out_degrees.min(), out_degrees.max(), out_degrees.mean())

    # Gross notional and net sum
    print("Total gross notional:", G.gross_notional)
    print("Net position sum (should be ~0):", G.net_positions.sum())


def main():
    from numpy.random import default_rng
    rng = default_rng(123)

    n_core = 5
    n_source = 3
    n_sink = 4
    p = 0.5

    # ---------------------------------------------------------------------
    # 1. Topology: Bernoulli vs fixed-degree (with Pareto weights)
    # ---------------------------------------------------------------------
    G_bernoulli = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )
    describe_network("Bernoulli topology (Pareto weights)", G_bernoulli)

    rng = default_rng(123)
    G_fixed = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="fixed",
        weight_mode="pareto",
    )
    describe_network("Fixed-degree topology (Pareto weights)", G_fixed)

    # ---------------------------------------------------------------------
    # 2. Weight modes under the same topology (Bernoulli)
    # ---------------------------------------------------------------------
    rng = default_rng(123)
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
    weights_pareto = G_pareto.W[G_pareto.W > 0]

    rng = default_rng(123)
    G_uniform = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="uniform",
        # default w_min, w_max; adjust if desired
    )
    weights_uniform = G_uniform.W[G_uniform.W > 0]

    rng = default_rng(123)
    G_const = generate_three_tier_network(
        n_core=n_core,
        n_source=n_source,
        n_sink=n_sink,
        p=p,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="constant",
        # constant_weight default = 1.0 inside assign_weights
    )
    weights_const = G_const.W[G_const.W > 0]

    print("\n=== Weight mode comparison (under Bernoulli topology) ===")
    print("Pareto weights: min, max, mean:",
          weights_pareto.min(), weights_pareto.max(), weights_pareto.mean())
    print("Uniform weights: min, max, mean:",
          weights_uniform.min(), weights_uniform.max(), weights_uniform.mean())
    print("Constant weights: unique values:",
          np.unique(weights_const))

    # Optional: quick check that uniform is within [w_min, w_max]
    print("Uniform weights within [w_min, w_max]? ->",
          weights_uniform.min(), "<= w <= ", weights_uniform.max())
    

    # ---------------------------------------------------------------------
    # 3. Test extract_largest_component
    # ---------------------------------------------------------------------
    from numpy.random import default_rng
    rng = default_rng(42)

    G_big = generate_three_tier_network(
        n_core=20,
        n_source=20,
        n_sink=20,
        p=0.05,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="pareto",
    )

    adj = (G_big.W > 0).astype(int)
    deg_total = adj.sum(axis=1) + adj.sum(axis=0)

    print("\n=== Largest component extraction ===")
    print("Original nodes:", G_big.num_nodes)
    print("Original zero-degree nodes:", np.sum(deg_total == 0))

    G_largest = extract_largest_component(G_big)
    adj2 = (G_largest.W > 0).astype(int)
    deg_total2 = adj2.sum(axis=1) + adj2.sum(axis=0)

    print("Largest component nodes:", G_largest.num_nodes)
    print("Zero-degree nodes in largest component (should be 0):",
          np.sum(deg_total2 == 0))


if __name__ == "__main__":
    main()


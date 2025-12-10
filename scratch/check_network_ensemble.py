import sys
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numpy.random import default_rng
from src.networks import generate_three_tier_network

try:
    import networkx as nx
except ImportError:
    nx = None


def run_ensemble(
    n_core=20,
    n_source=20,
    n_sink=20,
    p=0.1,
    degree_mode="bernoulli",
    weight_mode="pareto",
    n_samples=200,
):
    rng = default_rng(123)

    source_out_degrees = []
    sink_in_degrees = []
    core_out_degrees = []
    core_in_degrees = []

    density_sc = []   # source->core density
    density_cc = []   # core->core density
    density_ck = []   # core->sink density
    density_global = []

    gross_list = []
    all_weights = []

    largest_component_sizes = []
    isolated_counts = []

    for _ in range(n_samples):
        G = generate_three_tier_network(
            n_core=n_core,
            n_source=n_source,
            n_sink=n_sink,
            p=p,
            rng=rng,
            degree_mode=degree_mode,
            weight_mode=weight_mode,
        )

        W = G.W
        adj = (W > 0).astype(int)

        n = G.num_nodes
        ns, nc, nk = len(G.source_nodes), len(G.core_nodes), len(G.sink_nodes)

        # Degree stats
        out_deg = adj.sum(axis=1)
        in_deg = adj.sum(axis=0)

        source_out_degrees.extend(out_deg[G.source_nodes])
        sink_in_degrees.extend(in_deg[G.sink_nodes])
        core_out_degrees.extend(out_deg[G.core_nodes])
        core_in_degrees.extend(in_deg[G.core_nodes])

        # Block densities
        A_sc = adj[np.ix_(G.source_nodes, G.core_nodes)]
        A_cc = adj[np.ix_(G.core_nodes, G.core_nodes)].copy()
        np.fill_diagonal(A_cc, 0)  # exclude self-loops
        A_ck = adj[np.ix_(G.core_nodes, G.sink_nodes)]

        possible_sc = ns * nc
        possible_cc = nc * (nc - 1)
        possible_ck = nc * nk
        possible_global = n * (n - 1)

        edges_sc = A_sc.sum()
        edges_cc = A_cc.sum()
        edges_ck = A_ck.sum()
        edges_global = adj.sum() - np.trace(adj)

        density_sc.append(edges_sc / possible_sc if possible_sc > 0 else np.nan)
        density_cc.append(edges_cc / possible_cc if possible_cc > 0 else np.nan)
        density_ck.append(edges_ck / possible_ck if possible_ck > 0 else np.nan)
        density_global.append(edges_global / possible_global if possible_global > 0 else np.nan)

        # Weights
        weights = W[W > 0]
        if weights.size > 0:
            all_weights.append(weights)

        # Gross notional
        gross_list.append(G.gross_notional)

        # Connectivity (weakly)
        if nx is not None:
            H = nx.from_numpy_array(adj, create_using=nx.Graph)
            comps = list(nx.connected_components(H))
            sizes = [len(c) for c in comps]
            largest_component_sizes.append(max(sizes))
            isolated_counts.append(sum(1 for s in sizes if s == 1))

    all_weights = np.concatenate(all_weights) if all_weights else np.array([])

    return {
        "params": dict(
            n_core=n_core,
            n_source=n_source,
            n_sink=n_sink,
            p=p,
            degree_mode=degree_mode,
            weight_mode=weight_mode,
            n_samples=n_samples,
        ),
        "source_out": np.array(source_out_degrees),
        "sink_in": np.array(sink_in_degrees),
        "core_out": np.array(core_out_degrees),
        "core_in": np.array(core_in_degrees),
        "density_sc": np.array(density_sc),
        "density_cc": np.array(density_cc),
        "density_ck": np.array(density_ck),
        "density_global": np.array(density_global),
        "gross": np.array(gross_list),
        "weights": all_weights,
        "largest_component_sizes": np.array(largest_component_sizes)
            if largest_component_sizes else None,
        "isolated_counts": np.array(isolated_counts)
            if isolated_counts else None,
    }


def summarize_degrees_and_densities(stats):
    p = stats["params"]["p"]
    n_core = stats["params"]["n_core"]

    s_out = stats["source_out"]
    s_in = stats["sink_in"]
    c_out = stats["core_out"]
    c_in = stats["core_in"]

    dens_sc = stats["density_sc"]
    dens_cc = stats["density_cc"]
    dens_ck = stats["density_ck"]
    dens_global = stats["density_global"]

    isolates = stats["isolated_counts"]
    largest = stats["largest_component_sizes"]

    print("\n=== Ensemble summary: degrees and densities ===")
    print("Params:", stats["params"])

    print("\nSource out-degree:")
    print("  mean:", s_out.mean(), "  var:", s_out.var())
    print("  theo mean (Binomial):", p * n_core, "  theo var:", p * (1 - p) * n_core)

    print("\nSink in-degree:")
    print("  mean:", s_in.mean(), "  var:", s_in.var())
    print("  theo mean (Binomial):", p * n_core, "  theo var:", p * (1 - p) * n_core)

    print("\nCore out-degree (summary):")
    print("  mean:", c_out.mean(), "  var:", c_out.var())

    print("\nDensities (mean over ensemble):")
    print("  source->core:", dens_sc.mean())
    print("  core->core  :", dens_cc.mean())
    print("  core->sink  :", dens_ck.mean())
    print("  global      :", dens_global.mean())
    print("  target p    :", p)

    if isolates is not None:
        print("\nIsolated nodes per draw (weak components of size 1):")
        print("  mean:", isolates.mean(), "  min:", isolates.min(), "  max:", isolates.max())

    if largest is not None:
        print("\nLargest component size (weakly connected):")
        print("  mean:", largest.mean(), "  min:", largest.min(), "  max:", largest.max())


def summarize_weights(stats):
    w = stats["weights"]
    mode = stats["params"]["weight_mode"]
    print(f"\n=== Weight distribution summary (mode={mode}) ===")
    if w.size == 0:
        print("No edges, no weights.")
        return

    print("  count:", w.size)
    print("  min  :", w.min())
    print("  max  :", w.max())
    print("  mean :", w.mean())
    print("  std  :", w.std())
    print("  quantiles (50%, 90%, 99%):",
          np.quantile(w, [0.5, 0.9, 0.99]))


def main():
    # --------------------------------------------------------------
    # A) Compare degree statistics & densities: bernoulli vs fixed
    # --------------------------------------------------------------
    print("### DEGREE MODES: Bernoulli vs fixed (Pareto weights) ###")

    stats_bern = run_ensemble(
        n_core=20,
        n_source=20,
        n_sink=20,
        p=0.1,
        degree_mode="bernoulli",
        weight_mode="pareto",
        n_samples=200,
    )
    summarize_degrees_and_densities(stats_bern)

    stats_fixed = run_ensemble(
        n_core=20,
        n_source=20,
        n_sink=20,
        p=0.1,
        degree_mode="fixed",
        weight_mode="pareto",
        n_samples=200,
    )
    summarize_degrees_and_densities(stats_fixed)

    # --------------------------------------------------------------
    # B) Compare weight distributions: pareto vs uniform vs constant
    # --------------------------------------------------------------
    print("\n\n### WEIGHT MODES under Bernoulli topology ###")

    for weight_mode in ["pareto", "uniform", "constant"]:
        stats_w = run_ensemble(
            n_core=20,
            n_source=20,
            n_sink=20,
            p=0.1,
            degree_mode="bernoulli",
            weight_mode=weight_mode,
            n_samples=200,
        )
        summarize_weights(stats_w)

        # Also check that densities don’t change with weight_mode
        if weight_mode == "pareto":
            base_dens = stats_w["density_global"].mean()
        else:
            print("  global density mean:", stats_w["density_global"].mean(),
                  "(should be close to Pareto case:", base_dens, ")")


if __name__ == "__main__":
    main()

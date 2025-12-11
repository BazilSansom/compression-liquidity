# scratch/test_compression.py

import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.networks import generate_three_tier_network
from src.compression import (
    compress_BFF,
    compress_maxC,
    validate_conservative_compression,
    validate_full_conservative,
)


def check_net_preservation(G_orig, G_comp, tol: float = 1e-10) -> float:
    """
    Print and return the max deviation in net positions between two networks.
    """
    net_orig = G_orig.net_positions
    net_comp = G_comp.net_positions
    diff = np.abs(net_orig - net_comp)
    err = float(np.max(diff))
    err_print = 0.0 if err < tol else err
    print(f"Max net position deviation: {err_print}")
    return err


def main():
    rng = default_rng(42)

    G = generate_three_tier_network(
        n_core=10,
        n_source=10,
        n_sink=10,
        p=0.4,                  # a bit denser to get cycles
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="pareto",
        round_to=0.01,           # money-like weights
    )

    gross_before = float(G.W.sum())
    print(f"Gross before: {gross_before:.2f}")

    # --- BFF compression ---
    res_bff = compress_BFF(G)
    print("\n=== BFF compression ===")
    print("Gross after:", res_bff.gross_after)
    print("Savings abs:", res_bff.savings_abs)
    print("Savings frac:", res_bff.savings_frac)
    check_net_preservation(G, res_bff.compressed)

    # --- max-C compression (ORTools) ---
    res_maxC = compress_maxC(G, solver="ortools")
    print(f"\n=== max-C compression ({res_maxC.method}) ===")
    print("Gross after:", res_maxC.gross_after)
    print("Savings abs:", res_maxC.savings_abs)
    print("Savings frac:", res_maxC.savings_frac)
    check_net_preservation(G, res_maxC.compressed)

    # --- Happy-path explicit validation ---
    # These should all be silent if everything is correct.
    validate_conservative_compression(G, res_bff)
    validate_full_conservative(G, res_bff)

    validate_conservative_compression(G, res_maxC)
    validate_full_conservative(G, res_maxC)

    # --- Simple assertions for regression-style testing ---
    # Gross should not increase
    assert res_bff.gross_after <= gross_before + 1e-8
    assert res_maxC.gross_after <= gross_before + 1e-8

    # max-C should compress at least as much as BFF (or equal)
    assert res_maxC.gross_after <= res_bff.gross_after + 1e-8

    print("\nAll compression constraints and checks passed on valid outputs.")

    # =====================================================================
    #  DELIBERATE MISCHIEF: TEST THAT THE LIFEJACKETS INFLATE
    # =====================================================================

    print("\nNow deliberately breaking results to test validators...")

    # 1) Break net positions: tweak one edge in the compressed network
    broken_net = res_bff
    broken_net.compressed.W = broken_net.compressed.W.copy()
    broken_net.compressed.W[0, 1] += 1.0  # mess with one entry

    try:
        validate_conservative_compression(G, broken_net)
        print("ERROR: broken net positions were NOT detected!")
    except ValueError as e:
        print("OK: conservative validation failed on tampered nets as expected:")
        print("   ", e)

    # 2) Break acyclicity: add a tiny self-loop to create a directed cycle
    broken_cycle = res_bff
    broken_cycle.compressed.W = broken_cycle.compressed.W.copy()
    broken_cycle.compressed.W[0, 0] += 1.0  # add a self-loop

    try:
        validate_full_conservative(G, broken_cycle)
        print("ERROR: introduced cycle was NOT detected!")
    except ValueError as e:
        print("OK: full-conservative validation failed on cyclic result as expected:")
        print("   ", e)

    print("\nDeliberate-break tests completed.")


if __name__ == "__main__":
    main()

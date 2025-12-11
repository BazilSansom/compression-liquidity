import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numpy.random import default_rng
from src.networks import generate_three_tier_network
from src.compression import (
    compress_BFF,
    validate_conservative_compression,
)


def main():
    rng = default_rng(123)

    G = generate_three_tier_network(
        n_core=5,
        n_source=5,
        n_sink=5,
        p=0.1,
        rng=rng,
        degree_mode="bernoulli",
        weight_mode="pareto",
        round_to=0.01,
    )

    # Get a valid compression first
    res = compress_BFF(G)
    print("Validation on correct result:")
    validate_conservative_compression(G, res)
    print("  -> passed\n")

    # --- 1. Break net positions ---
    print("Validation after breaking net positions (should FAIL):")
    bad_res = res
    # add a tiny perturbation to one row of W
    bad_res.compressed.W[0, 1] += 1.0

    try:
        validate_conservative_compression(G, bad_res)
    except ValueError as e:
        print("  Caught expected ValueError:")
        print("   ", e)
    else:
        print("  ERROR: validation did NOT catch broken net positions!")

    # --- 2. Create a new edge that wasn't there before ---
    print("\nValidation after creating new edge (should FAIL):")
    # find a zero entry in original that is positive in compressed
    W0 = G.W
    W1 = bad_res.compressed.W
    # Force an edge where before there was none
    i, j = 0, 2
    if W0[i, j] == 0.0:
        W1[i, j] = 0.5
    else:
        # find some other zero location
        zero_positions = np.argwhere(W0 == 0.0)
        i, j = zero_positions[0]
        W1[i, j] = 0.5

    try:
        validate_conservative_compression(G, bad_res)
    except ValueError as e:
        print("  Caught expected ValueError:")
        print("   ", e)
    else:
        print("  ERROR: validation did NOT catch new edges!")


if __name__ == "__main__":
    main()

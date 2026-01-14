# src/rng.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RNGStreams:
    # Network
    net_topology: np.random.Generator
    net_weights: np.random.Generator

    # Shocks (kept for backwards-compat + explicit substreams)
    shock_shape: np.random.Generator   # legacy/general-purpose
    shock_cal: np.random.Generator     # exp3 calibration shapes
    shock_mc: np.random.Generator      # exp3 evaluation MC shapes


def make_streams(master_seed: int, draw_id: int) -> RNGStreams:
    """
    Deterministically create independent RNG streams for a given Monte Carlo draw.

    Structure:
      draw
        ├── network
        │     ├── topology
        │     └── weights
        └── shock
              ├── shape (legacy)
              ├── cal
              └── mc
    """
    root = np.random.SeedSequence([master_seed, draw_id])

    # Two independent top-level branches
    ss_network, ss_shock = root.spawn(2)

    # Network substreams
    ss_topology, ss_weights = ss_network.spawn(2)

    # Shock substreams
    ss_shape, ss_cal, ss_mc = ss_shock.spawn(3)

    return RNGStreams(
        net_topology=np.random.default_rng(ss_topology),
        net_weights=np.random.default_rng(ss_weights),
        shock_shape=np.random.default_rng(ss_shape),
        shock_cal=np.random.default_rng(ss_cal),
        shock_mc=np.random.default_rng(ss_mc),
    )

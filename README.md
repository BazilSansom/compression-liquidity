# compression-liquidity
Temporary private repository for finalising code for the compression–liquidity paper.

# Compression & Liquidity Simulation

Code for the numerical experiments in the paper:

**The consequences of portfolio compression for systemic liquidity risk**  
*Working title*

This repository contains a modular and reproducible implementation of the  
simulation environment, compression algorithms, liquidity shock model,  
and ERLS experiments used in the paper.

The design separates:

- a **core reusable layer** (`src/`), containing the stable abstractions and algorithms  
  (network representation, compression, payments/FPA, shocks, buffers, ERLS), and
- **experiment scripts** (`experiments/`), which assemble the components to reproduce  
  the figures in the paper.

---


## 🗂 Repository Structure



```
src/                     ← core reusable library code
    networks.py          ← PaymentNetwork object; random network generator
    compression.py       ← BFF and max-C compression algorithms
    simulation.py        ← Full Payment Algorithm (FPA)
    shocks.py            ← intraday liquidity shock model & generators
    buffers.py           ← buffer construction and rescaling rules
    erls.py              ← Equal Risk Liquidity Savings (ERLS) routines
    metrics.py           ← derived metrics (optional)
    __init__.py

experiments/             ← experiment scripts for paper figures
notebooks/               ← exploratory notebooks
tests/                   ← unit tests (to be extended)

```

The `src/` directory is the reusable, stable part of the codebase that we get right once and is used by, but independent from experiments.

---

## Core Abstractions and Intended API

The model follows the conceptual flow used in the paper: Buffers b → Intraday shocks ξ → Available liquidity b - ξ → VM payment obligations → Full Payment Algorithm (FPA) → Liquidity shortfall L → ERLS computation.

The model contains two sources of randomness:
- the randomly generated OTC network (structure and notionals),
- the intraday liquidity shocks ξ applied to buffers.

VM obligations themselves are fixed by the network and shock scenarios we consider by varying shock size not random draws.


### PaymentNetwork

A minimal structure representing the network of VM payment obligations:

- `W[i, j]` = VM obligation from node *i* to node *j*  
- `node_types[i]` ∈ { "core", "source", "sink" }

Random networks are used to understand typical behaviour across ensembles of possible OTC configurations.

The network is the only object modified by compression.

---

### Network generation

`generate_three_tier_network(...)`  
Constructs synthetic OTC-style networks with a core–periphery structure  
and source/sink nodes, consistent with the model used in the paper.

---

### Compression algorithms

Compression produces a new network that preserves net positions but reduces gross exposure.

- `compress_maxC(network) → (compressed_network, stats)`
- `compress_BFF(network) → (compressed_network, stats)`

- **max-C**: implements full conservative compression by maximising circulation  
  (via CDFD), eliminating redundant cyclic exposures.
- **BFF**: implements alternative full concervative compression using Balanced Flow Forwarding algorithm.

---

### Shock model — intraday liquidity drains (not VM calls)

Shocks ξ represent **intraday liquidity drains** that occur *before* the VM settlement window.

These shocks:

- Reduce buffers available to meet VM obligations,
- Are one of the two **sources of randomness in the model** (along with network structure),  
- Are shared between pre- and post-compression networks to ensure comparability.

A shock model (following paper) a Gaussian copula with 

- ρ - correlation parameter 
- γ - magnitude parameter truncated to [0,γ_max]
- γ_max set proportional to median VM shocks


```
ShockModel(ρ=..., γ=...)
draw_shock(rng, network, shock_model) → np.ndarray
```
Each call produces a vector ξ such that:

`available_liquidity[i] = buffers[i] - ξ[i]`

This vector is fed directly into the Full Payment Algorithm.

---

### Payment dynamics (FPA)


`full_payment_algorithm(network, buffers, shock)`

Implements the **Full Payment Algorithm** (FPA), which clears VM obligations  
subject to liquidity constraints:

1. VM obligations are fixed by the network.  
2. Available liquidity is `buffers − shock`.  
3. Payments propagate through the system until no further payments are feasible.  
4. The output is the liquidity shortfall vector `L_i ≥ 0`.

This is the core mechanical engine underlying all experiments.

---

### Buffers, rescaling rules, and ERLS

There are two types of rescaling rules used in the paper:

- **Planner rule** (same-shape buffers):  
  all buffers are scaled proportionally.
- **Behavioural rule**:  
  only “flexible” nodes scale their buffers proportionally;  
  “inflexible” nodes keep their buffers unchanged.

These are implemented in:

```
initial_buffers(...)
planner_rescale_buffers(...)
behavioural_rescaling_proportional(...)
```

---

### ERLS — Equal Risk Liquidity Savings

ERLS measures the **maximum proportional reduction in liquidity buffers** allowed  
after compression **while keeping liquidity risk constant**.

Formally:

Let:
- `G_before` = uncompressed network  
- `G_after`  = compressed network  
- `b_before` = initial buffers  
- `ξ`        = the *same* intraday shock realisation  
- `L(G, b, ξ)` = liquidity shortfall from FPA  

Define ERLS(ρ) under a scaling rule ρ (planner or behavioural) as:

ERLS = max λ ∈ [0,1] such that L(G_after, ρ(b_before, λ), ξ) ≤ L(G_before, b_before, ξ)


Thus ERLS quantifies the **liquidity savings unlocked by compression**  
without increasing systemic liquidity risk.

The ERLS module provides:

`compute_erls(G_before, G_after, b_before, shock, scaling_rule)`

implemented via binary search over λ.

---

## Development Plan

1. **Network layer**
   - Implement random network generator
   - Add tests for constraints etc

2. **Compression algorithms**
   - Implement BFF and max-C
   - Checks: net positions preserved, gross exposure reduced etc.

3. **Payment dynamics**
   - Implement FPA
   - Add small-network sanity tests

4. **Shock model**
   - Implement intraday-drain generator
   - Add tests

5. **ERLS and buffer-scaling rules**
   - Planner search
   - Behavioural rule
   - Comparisons under identical shock scenarios

6. **Experiment scripts**
   - Implement experiments 1–3
   - Reproduce all figures in the paper

---

## Collaboration Rules

To keep the codebase coherent and reproducible:

### 1. Please do **NOT** push directly to `main`.

Instead:

- Create a feature branch:  
  `git checkout -b feature-name`
- Commit and push your branch:  
  `git push -u origin feature-name`
- Open a **Pull Request** for review and merging.

### 2. All merges into `main` go through Pull Requests.

### 3. Resolve all review comments before merging.

### 4. Keep `src/` stable.

Experiment-specific code should not leak into the reusable library once modules are finalised.

---

## Example Usage (illustrative)

```python
import numpy as np

from src.networks import generate_three_tier_network
from src.compression import compress_maxC
from src.buffers import initial_buffers
from src.shocks import ShockModel, draw_shock
from src.simulation import full_payment_algorithm
from src.erls import compute_erls
from src.buffers import planner_rescale_buffers

rng = np.random.default_rng(42)

# 1. Generate network
G = generate_three_tier_network(n_core=10, n_source=20, n_sink=20, rng=rng)

# 2. Initialise buffers
b0 = initial_buffers(G, level=1.0)

# 3. Sample intraday liquidity shock
shock_model = ShockModel(sigma=0.1, m=3.0)
xi = draw_shock(rng, G, shock_model)

# 4. Compress the network
G_comp, stats = compress_maxC(G)

# 5. Compute ERLS under planner scaling rule
erls = compute_erls(G_before=G, G_after=G_comp, b_before=b0, shock=xi,
                    scaling_rule=planner_rescale_buffers)

print("ERLS (planner):", erls)
```

This example demonstrates how network generation, compression, shocks, payments,
and ERLS knit together.


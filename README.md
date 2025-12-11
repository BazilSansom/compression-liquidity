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


# 🗂 Repository Structure

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

# Core Abstractions and Intended API

The model follows the conceptual flow used in the paper: Buffers b → Intraday shocks ξ → Available liquidity b - ξ → VM payment obligations → Full Payment Algorithm (FPA) → Liquidity shortfall L → ERLS computation.

The model contains two sources of randomness:
- the randomly generated OTC network (structure and notionals),
- the intraday liquidity shocks ξ applied to buffers.

VM obligations themselves are fixed by the network and shock scenarios we consider by varying shock size not random draws.


## 🕸️ Network Generation (`src/networks.py`)

The module `networks.py` provides a clean and flexible interface for generating synthetic OTC derivatives networks consistent with the three-tier “bow-tie” structure documented in the literature (e.g. D’Errico & Roukny 2017; Craig & von Peter 2014).

### PaymentNetwork

All networks are represented by the `PaymentNetwork` dataclass:

```python
@dataclass
class PaymentNetwork:
    W: np.ndarray              # weighted adjacency: obligations from i → j
    source_nodes: List[int]    # indices of source-tier nodes
    core_nodes: List[int]      # indices of dealer core
    sink_nodes: List[int]      # indices of sink-tier nodes
```

Convenience properties:

- `num_nodes`
- `gross_notional`
- `net_positions = outflows – inflows`

### Three–tier topology

Networks are generated using:

```python
generate_three_tier_network(
    n_core,
    n_source,
    n_sink,
    p,
    *,
    degree_mode="bernoulli",
    weight_mode="pareto",
    rng=None,
    ...
)
```

This creates a directed network with:

- **source → core** links
- **core ↔ core** ER links
- **core → sink** links

**Topology modes**

`degree_mode="bernoulli"` **(default)**

Each potential edge is present independently with probability $p$:
- source out-degree ∼Binomial($𝑛_{core},p$)
- sink in-degree ∼Binomial($𝑛_{core},p$)
- core–core forms a directed ER graph

This matches:
- The model described in our current draft of the paper
- D’Errico & Roukny (2017)
- Standard systemic-risk models using ER random graphs

`degree_mode="fixed"` **(optional)**
Each source/sink connects to approximately $pn_{core}$ dealers (degree-regular periphery), reproducing behaviour in earlier code versions.
Useful for robustness checks.

**Edge-weight options**

Weights can be generated with:
- **Heavy-tailed Pareto** (default)
- **Uniform** baseline
- **Constant** weights

```python
weight_mode="pareto" | "uniform" | "constant"
```

The default heavy-tailed Pareto weights reflect the empirical concentration of bilateral derivatives exposures and match the modelling choice used in our current draft of the paper and in D’Errico & Roukny (2017).

Uniform and constant weight modes are included as simple robustness baselines.

Optional parameters:
- `alpha_weights` – Pareto tail exponent
- `scale_weights` – global scaling
- `round_to` – round weights to nearest multiple (e.g. 0.01)


### Largest Component Extraction

Some random networks may contain isolated nodes or small disconnected components. A helper function is provided:

```python
from src.networks import extract_largest_component

G_clean = extract_largest_component(G)
```

This:
- treats edges as undirected for connectivity purposes,
- selects the largest weakly connected component,
- drops inactive nodes (degree 0),
- remaps tier indices correctly.

Useful when focusing on the “active” OTC market.


### Example Usage

```python
import numpy as np
from src.networks import generate_three_tier_network, extract_largest_component

rng = np.random.default_rng(42)

# Generate a Bernoulli topology with Pareto weights
G = generate_three_tier_network(
    n_core=20,
    n_source=20,
    n_sink=20,
    p=0.1,
    rng=rng,
    degree_mode="bernoulli",
    weight_mode="pareto",
)

print("Gross notional:", G.gross_notional)
print("Net position sum:", G.net_positions.sum())  # ≈ 0

# Optionally focus on the active market
G_largest = extract_largest_component(G)
print("Nodes in largest component:", G_largest.num_nodes)
```

---

## 🧩 Compression algorithms (`src/compression.py`)

This module implements full conservative portfolio compression for PaymentNetwork objects using the circular flow decomposition (CDFD) (Homs Dones et al. 2025). It provides:
- BFF compression — uses the Balanced Flow Forwarding algorithm to remove redundent possitions leaving no cycles.
- max-C compression — finds the maximum possible reduction in gross notional while preserving net positions (leaves no cycles).
- Validation utilities — ensuring that compression outputs satisfy all required constraints.

All compression routines return a CompressionResult dataclass containing the compressed network, savings statistics, and the CDFD circular ($C$) and directional ($D$) components.

**✔️ What Compression Guarantees**

A valid conservative compression must satisfy:
1. **Net positions preserved** $outflow_{i}-inflow_{i}$ unchanged for all $i$
2. **No new counterparties** Support of the compressed network must be a subset of the original support.
3. **No negative edges**
4. **Gross notional does not increase**

A valid full conservative compression must additionally satisfy:

5. **Acyclic directional part** (compressed network is a DAG)

These conditions are formally checked by:
- `validate_conservative_compression(...)`
- `validate_full_conservative(...)`
  
These are called automatically inside `compress_BFF` and `compress_maxC` unless disabled.

**⚙️ API Summary**
`compress_BFF(G, tol_zero=1e-12, require_conservative=True, require_full_conservative=True)`

- Performs compression using the Balanced Flow Forwarding (BFF) method.
- Returns `CompressionResult`.

`compress_maxC(G, solver="ortools", ...)`

- Computes maximal compression, i.e. minimises notional
- Returns CompressionResult.

Available solvers:
- `"ortools"` (default): Integer min-cost flow, robust on Apple Silicon, suitable for money-like data.
- `"pulp"`: PuLP + CBC solver. Not recommended on Apple Silicon unless CBC is separately installed.

**📦 `CompressionResult`**

```python
@dataclass
class CompressionResult:
    compressed: PaymentNetwork       # compressed network (directional part)
    method: str                      # "BFF", "maxC_ortools", ...
    gross_before: float
    gross_after: float
    savings_abs: float
    savings_frac: float
    C_circular: csr_array | None     # circular component
    D_directional: csr_array         # directional component
    meta: dict                       # solver diagnostics

```

**🔍 Validation Functions (Safety Checks)**

`validate_conservative_compression(G_before, result, tol=1e-10)`

Ensures the solution satisfies:
- net positions preserved
- no new edges
- non-negative weights
- gross notional does not increase

Raises ValueError on violations.

`validate_full_conservative(G_before, result, tol=1e-10)`

Ensures:
- conservative compression (above)
- compressed network is acyclic

Raises ValueError on violations.


### Example usage

```python
import numpy as np
from numpy.random import default_rng

from src.networks import generate_three_tier_network
from src.compression import compress_BFF, compress_maxC

rng = default_rng(123)

# 1. Generate a synthetic OTC-style network
G = generate_three_tier_network(
    n_core=10,
    n_source=5,
    n_sink=5,
    p=0.05,
    rng=rng,
    degree_mode="bernoulli",
    weight_mode="pareto",
    round_to=0.01,
)

# 2. Apply BFF-based compression (full conservative by default)
res_bff = compress_BFF(G)

print("BFF gross before:", res_bff.gross_before)
print("BFF gross after :", res_bff.gross_after)
print("BFF savings (%) :", 100 * res_bff.savings_frac)

# 3. Apply maximal compression (min-cost flow via OR-Tools)
res_maxC = compress_maxC(G, solver="ortools")

print("max-C gross after:", res_maxC.gross_after)
print("max-C savings (%) :", 100 * res_maxC.savings_frac)

# The compressed payment network (directional part)
G_comp = res_maxC.compressed
print("Net positions preserved?", np.allclose(G.net_positions, G_comp.net_positions))
```


---

## 📘 `simulation.py` — Full Payment Algorithm (FPA)

The `simulation.py` module implements the Full Payment Algorithm (FPA) as described in Bardoscia et al. (2019).
FPA iteratively settles obligations on a directed liabilities network given available liquidity.
It forms the core payment-clearing procedure used throughout this project.

🚀 `run_fpa(L, e0)` — compute clearing payments and shortfalls

```python
from src.simulation import run_fpa
```

| Parameter      | Description                                                                          |
| -------------- | ------------------------------------------------------------------------------------ |
| `L`            | (N \times N) *liabilities matrix*: `L[i, j]` = obligation from node *i* to node *j*. |
| `e0`           | Length-(N) array of initial cash balances (b(0)).                                    |
| `rel_tol`      | Numerical tolerance for liquidity checks (default `1e-8`).                           |
| `return_paths` | If `True`, returns the full sequence of cash/payment vectors.                        |

Outputs: `FPAResult`

`run_fpa` returns a dataclass:

```python
@dataclass
class FPAResult:
    residual_obligations: np.ndarray   # l_ss
    final_cash: np.ndarray             # b_ss
    final_payments: np.ndarray         # payments in final iteration
    residual_matrix: np.ndarray        # Pi * l_ss
    iterations: int                    # number of updates after t=0
    first_round_residual: np.ndarray   # l(1)
    shortfall: np.ndarray              # max(0, l_ss - b_ss)
    aggregate_shortfall: float         # sum(shortfall)
    cash_path: list[np.ndarray] | None
    payment_path: list[np.ndarray] | None
    indicator_path: list[np.ndarray] | None
```

Key result attributes:
- `residual_obligations`: unpaid obligations at termination
- `final_cash`: steady-state cash
- `shortfall`: node-level liquidity shortfalls (equation 4 in the paper)
- `aggregate_shortfall`: system-wide shortfall measure $R$ (equation ? in the paper)

All vectors are $𝑁×1$ column vectors.

### ✨ Example

```python
import numpy as np
from src.simulation import run_fpa

# Simple 3-node cycle: A -> B -> C -> A
Pi = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], float)

pbar = np.array([10.0, 7.0, 5.0])    # total obligations
L = np.diag(pbar) @ Pi               # liabilities matrix

e0 = np.array([12.0, 4.0, 3.0])      # initial liquidity

result = run_fpa(L, e0)

print(result.final_cash)
print(result.shortfall)
print(result.aggregate_shortfall)
```

Expected output:

```python
final_cash =
[[7.],
 [7.],
 [5.]]
shortfall =
[[0.],
 [0.],
 [0.]]
aggregate_shortfall = 0.0
```

This reproduces Example 1 from Bardoscia et al. (2019).

### 🧪 Testing and correctness

The implementation is validated against the canonical worked examples in the original FPA paper.

Tests live under:
`tests/test_fpa_paper_examples.py`


---

## Shock model — intraday liquidity drains (not VM calls)

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


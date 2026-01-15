# compression-liquidity

API for the compression–liquidity paper 'Portfolio compression, settlement liquidity, and systemic risk' (working title) with Yijie Zhou, Marc Homs Dones, Robert Mackay.

## Reproducing the paper results

This repository contains the code used to generate the simulation artifacts, figures, tables, and LaTeX number/parameter exports for the paper.

### Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install runtime and dev dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**Run the full test suite:**
```bash
pytest -q
```

**Reproduce all results (Experiments 1–3)**
The main entrypoint is:
```bash
python -m paper.make_all
```
This runs Experiments 1–3, builds figures/tables, and writes LaTeX exports. By default it runs in **smoke mode** `(--mode small)` and tags outputs as `paper-smoke` (so smoke runs do not overwrite “paper” runs).

**Smoke run (fast)**
Runs a reduced version of Experiments 1–3 and writes outputs under the `paper-smoke` tag:
```bash
python -m paper.make_all --mode small
```
You can also explicitly set a tag:
```bash
python -m paper.make_all --mode small --tag my-smoke-tag
```
**Full run (paper defaults)**
Runs the full default seed sets and writes outputs under the `paper` tag:
```bash
python -m paper.make_all --mode full
```
(Optional) set a custom tag:
```bash
python -m paper.make_all --mode full --tag paper-v1
```
**Build figures/tables without re-running simulations**
You can rebuild figures/tables from existing artifacts (e.g. for working on polishing figure labels / table formatting):
```bash
python -m paper.make_all --figures-only
```
By default, `--figures-only` rebuilds from the latest artifacts tagged `paper`. To rebuild from a different tag:
```bash
python -m paper.make_all --figures-only --artifact-tag paper-smoke
```
**Useful flags**
- `skip-figures`: run simulations only (no figure/table build)
- `skip-export`: skip LaTeX exporters


**Outputs**

All generated outputs are written under:
- `outputs/paper/artifacts/exp{1,2,3}/<run_id>/` (raw simulation artifacts per run)
- `outputs/paper/figures/exp{1,2,3}/<run_id>/` (figures)
- `outputs/paper/tables/exp{1,2,3}/<run_id>/` (tables and regression outputs)

Paper-facing LaTeX exports (maps numbers to latex macros and provides single source of truth for latex draft):
- `outputs/paper/paper_inputs.tex` (model parameters used in the paper)
- `outputs/paper/paper_numbers.tex` (paper numbers derived from Exp 1-3 summary)
- `outputs/paper/exp3_summary.json` (canonical Exp3 summary used by the exporter)


# 🗂 Repository Structure

```
src/                     ← core library code (network generation, compression, shock generation, settlment dynamics etc.)
paper/                   ← paper-specific specs, runners, figure/table builders, and LaTeX exporters
  specs/                 ← defines experiment specifications and default seed sets (the paper’s “configuration surface”).
  runners/               ← runs simulations and writes raw artifacts to `outputs/paper/artifacts/...`.
  figures/               ← builds figures/tables from an artifact directory (no simulation).
  build/                 ← orchestrates full builds and exports LaTeX macros (single source of truth for numbers/parameters).
tests/                   ← unit tests

```
### Separation of concerns (`src/` vs `paper/` and within paper pipeline)

- `src/` is the reusable library: core data structures and algorithms (networks, compression, FPA clearing, shocks, buffers, ERLS, utilities). It is intended to be stable and importable outside the paper pipeline.
- `paper/` is the reproducible paper pipeline: experiment specs, runners that generate artifacts, figure/table builders, and LaTeX exporters.The `paper/` package is organised to keep the build pipeline reproducible and easy to iterate on.

Rule: `paper/` may import `src/`, but `src/` must not import `paper/`.


## `src/` toolbox documentation

Detailed documentation of the reusable library (`src/`) lives in `src/README.md`.


## Contributing

Please open a pull request for changes (no direct pushes to `main`).
See `CONTRIBUTING.md` for workflow, tests, and style conventions.


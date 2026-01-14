# Contributing

This repository supports an in-progress research paper. Until publication, the API may evolve, but we aim to keep changes **reproducible**, **reviewable**, and **well-tested**.

## Workflow (required)

### 1) No direct pushes to `main`
Please work on a feature branch:

```bash
git checkout -b feature/<short-description>
```

Commit locally and push your branch:

```bash
git push -u origin feature/<short-description>
```

Then open a **Pull Request** into main.

> Note: `main` is protected and requires PRs + CI to pass.

### 2) Pull Requests only

All merges into `main` happen via PR. Please resolve review comments before merging.

### 3) Tests must pass
Run the test suite before opening a PR:
```bash
pytest -q
```

## Separation of concerns: `src/` vs `paper/`

`src/` **= reusable library (stable)**

`src/` contains the core model components (networks, compression, shocks, FPA, ERLS, utilities).
Changes here should be treated as API changes and should be done carefully:

- keep changes small and well-scoped
- update unit tests in tests/
- update docstrings and any public-facing documentation
- avoid experiment-specific “glue” code creeping into src/

If a change is purely for producing a figure/table or paper artifact, it likely belongs in paper/.

`paper/` **= paper pipeline (allowed to evolve faster)**

`paper/` contains experiment specs, runners, figure/table builders, and build/export scripts.
It is the reproducible pipeline used to generate paper artifacts.

## Output policy

Generated outputs live under `outputs/` and should not be committed.

- smoke runs use the `paper-smoke` tag by default
- full runs use the `paper` tag by default

(See `README.md` for the standard commands.)
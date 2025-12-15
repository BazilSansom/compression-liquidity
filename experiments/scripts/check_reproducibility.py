# experiments/scripts/check_reproducibility.py
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataclasses import replace

from experiments.scripts.paper_config import (
    FIG_ROOT,
    N_DRAWS,
    PAPER_BASE_SPEC,
    THETA_FIXED_BUFFERS,
)
from experiments.runners.shortfall_vs_lambda_core import run_shortfall_vs_lambda_optionA
from experiments.runners.erls_vs_lambda_core import run_erls_vs_lambda_optionA


# -----------------------
# helpers
# -----------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _compare_csv(a: Path, b: Path, *, sort_cols: Iterable[str]) -> None:
    """
    Compare two CSVs robustly:
      - sort rows by sort_cols
      - compare all numeric columns with allclose
      - compare all non-numeric columns with exact match
    """
    df1 = pd.read_csv(a)
    df2 = pd.read_csv(b)

    # make sure columns align
    if set(df1.columns) != set(df2.columns):
        raise AssertionError(f"Column mismatch:\n{a}: {df1.columns}\n{b}: {df2.columns}")

    df1 = df1[df2.columns]  # same order
    df1 = df1.sort_values(list(sort_cols)).reset_index(drop=True)
    df2 = df2.sort_values(list(sort_cols)).reset_index(drop=True)

    if len(df1) != len(df2):
        raise AssertionError(f"Row count mismatch: {a} has {len(df1)}, {b} has {len(df2)}")

    # Compare
    for c in df1.columns:
        s1, s2 = df1[c], df2[c]
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            if not (s1.fillna(0.0).astype(float).sub(s2.fillna(0.0).astype(float)).abs().max() <= 1e-12):
                # fall back to allclose-like check
                import numpy as np

                if not np.allclose(s1.to_numpy(dtype=float), s2.to_numpy(dtype=float), rtol=0.0, atol=1e-12, equal_nan=True):
                    raise AssertionError(f"Numeric column differs: {c}")
        else:
            if not s1.fillna("").astype(str).equals(s2.fillna("").astype(str)):
                raise AssertionError(f"Non-numeric column differs: {c}")

    print(f"✓ CSVs match (robust compare): {a.name}")


# -----------------------
# main reproducibility run
# -----------------------
def _run_once(out_root: Path) -> dict[str, Path]:
    """
    Run Exp1 + Exp2 once, writing CSVs into out_root.
    Returns a dict of named outputs.
    """
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # -----------------------
    # Experiment 1 (fixed buffers)
    # -----------------------
    theta = THETA_FIXED_BUFFERS
    theta_tag = f"{theta:g}"
    spec1 = replace(PAPER_BASE_SPEC, name=f"{PAPER_BASE_SPEC.name}_exp1_fixed_theta{theta_tag}")

    exp1_dir = out_root / "exp1"
    exp1_dir.mkdir(parents=True, exist_ok=True)

    rows1_bff = run_shortfall_vs_lambda_optionA(
        spec1,
        n_draws=N_DRAWS,
        buffer_theta=theta,
        compression_method="bff",
        out_csv=None,
        plot=False,
    )
    rows1_maxc = run_shortfall_vs_lambda_optionA(
        spec1,
        n_draws=N_DRAWS,
        buffer_theta=theta,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=None,
        plot=False,
    )

    p1_bff = exp1_dir / "exp1_bff.csv"
    p1_maxc = exp1_dir / "exp1_maxc.csv"
    _write_csv(rows1_bff, p1_bff)
    _write_csv(rows1_maxc, p1_maxc)
    outputs["exp1_bff"] = p1_bff
    outputs["exp1_maxc"] = p1_maxc

    # -----------------------
    # Experiment 2 (behavioural ERLS)
    # -----------------------
    spec2 = replace(PAPER_BASE_SPEC, name=f"{PAPER_BASE_SPEC.name}_exp2_behavioural", buffer_mode="behavioural")

    exp2_dir = out_root / "exp2"
    exp2_dir.mkdir(parents=True, exist_ok=True)

    rows2_bff = run_erls_vs_lambda_optionA(
        spec2,
        n_draws=N_DRAWS,
        compression_method="bff",
        out_csv=None,
        plot=False,
    )
    rows2_maxc = run_erls_vs_lambda_optionA(
        spec2,
        n_draws=N_DRAWS,
        compression_method="maxc",
        compression_solver="ortools",
        out_csv=None,
        plot=False,
    )

    p2_bff = exp2_dir / "exp2_bff.csv"
    p2_maxc = exp2_dir / "exp2_maxc.csv"
    _write_csv(rows2_bff, p2_bff)
    _write_csv(rows2_maxc, p2_maxc)
    outputs["exp2_bff"] = p2_bff
    outputs["exp2_maxc"] = p2_maxc

    return outputs


def main() -> None:
    """
    Runs Exp1+Exp2 twice and checks CSV reproducibility.

    Usage:
        python experiments/scripts/check_reproducibility.py
    """
    # write under figures/paper/_repro_check (separate from paper outputs)
    repro_root = FIG_ROOT / "_repro_check"
    run1_root = repro_root / "run1"
    run2_root = repro_root / "run2"

    print(f"Running reproducibility check under: {repro_root}")
    outs1 = _run_once(run1_root)
    outs2 = _run_once(run2_root)

    # Compare CSVs (robust compare + hash)
    comparisons = [
        ("exp1_bff", ("draw", "lam")),
        ("exp1_maxc", ("draw", "lam")),
        ("exp2_bff", ("draw", "lam")),
        ("exp2_maxc", ("draw", "lam")),
    ]

    for key, sort_cols in comparisons:
        a = outs1[key]
        b = outs2[key]
        h1 = _sha256(a)
        h2 = _sha256(b)
        print(f"{key}: sha256 run1={h1[:12]} run2={h2[:12]}")
        if h1 == h2:
            print(f"✓ Hash match: {a.name}")
        else:
            print(f"⚠ Hash differs for {a.name}; running robust compare...")
            _compare_csv(a, b, sort_cols=sort_cols)

    print("\nAll reproducibility checks passed.")


if __name__ == "__main__":
    main()

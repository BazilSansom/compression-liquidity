# tests/test_erls.py
from __future__ import annotations

import numpy as np
import pytest

from src.erls import xi_from_uniform_base, compute_erls_zero_shortfall

from tests.fixtures import (
    bardoscia_example1,
    bardoscia_example2,
    bardoscia_example3,
)


def _as_col(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


# -------------------------
# Unit tests: xi construction
# -------------------------

@pytest.mark.unit
def test_xi_from_uniform_base_requires_correct_shape():
    V_ref = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)

    # Wrong shape: (N,) instead of (N,1) is accepted by reshape, so test a truly wrong length:
    U_wrong = np.array([0.1, 0.2, 0.3], dtype=float)  # length 3, but N=2
    with pytest.raises(ValueError):
        xi_from_uniform_base(V_ref=V_ref, U=U_wrong, lam=0.1, scale="row_sum")


@pytest.mark.unit
@pytest.mark.parametrize("scale", ["row_sum", "col_sum", "total"])
def test_xi_from_uniform_base_scales_linearly_in_lam(scale: str):
    V_ref = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    N = V_ref.shape[0]
    U = np.full((N, 1), 0.5, dtype=float)

    xi1 = xi_from_uniform_base(V_ref=V_ref, U=U, lam=0.2, scale=scale)
    xi2 = xi_from_uniform_base(V_ref=V_ref, U=U, lam=0.4, scale=scale)

    assert xi1.shape == (N, 1)
    assert xi2.shape == (N, 1)
    assert np.allclose(xi2, 2.0 * xi1, atol=1e-12, rtol=0.0)


@pytest.mark.unit
def test_xi_from_uniform_base_row_sum_matches_definition():
    V_ref = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    U = np.array([[1.0], [0.5], [0.0]], dtype=float)
    lam = 0.1

    s = V_ref.sum(axis=1).reshape(-1, 1)
    xi = xi_from_uniform_base(V_ref=V_ref, U=U, lam=lam, scale="row_sum")

    assert np.allclose(xi, lam * s * U, atol=1e-12, rtol=0.0)


# ---------------------------------
# Integration tests: ERLS computation
# ---------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_fn",
    [bardoscia_example1, bardoscia_example2, bardoscia_example3],
)
def test_compute_erls_zero_shortfall_hits_target_and_returns_sane_values(fixture_fn):
    V, _e0, *_ = fixture_fn()
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    # Use deterministic U_base in [0,1] with nonzero entries
    U_base = np.full((N, 1), 0.5, dtype=float)
    lam = 0.2

    # Use V_tilde = V (identity case)
    res = compute_erls_zero_shortfall(
        V=V,
        V_tilde=V,
        U_base=U_base,
        lam=lam,
        xi_scale="row_sum",
        buffer_mode="fixed_shape",
        rel_tol_fpa=1e-10,
    )

    # Should hit (approximately) zero shortfall both pre and post
    assert res.R_pre <= 1e-8
    assert res.R_post <= 1e-8

    # Buffers are nonnegative totals
    assert res.B_pre >= -1e-12
    assert res.B_post >= -1e-12

    # ERLS is defined as 1 - B_post/B_pre when B_pre>0, else 0.
    # In identity case (V_tilde=V) under fixed_shape, should be ~0.
    assert res.erls == pytest.approx(0.0, abs=1e-8)

    # If alpha_pre is positive, kappa should be ~1 in the identity case.
    if res.alpha_pre > 0:
        assert res.kappa == pytest.approx(1.0, abs=1e-8)


@pytest.mark.integration
def test_compute_erls_zero_shortfall_behavioural_identity_case_gives_zero_erls():
    # Behavioural mode recomputes shape post-compression; if V_tilde == V, still should match.
    V, _e0, *_ = bardoscia_example2()
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    U_base = np.full((N, 1), 0.7, dtype=float)
    lam = 0.15

    res = compute_erls_zero_shortfall(
        V=V,
        V_tilde=V,
        U_base=U_base,
        lam=lam,
        xi_scale="row_sum",
        buffer_mode="behavioural",
        rel_tol_fpa=1e-10,
    )

    assert res.R_pre <= 1e-8
    assert res.R_post <= 1e-8
    assert res.erls == pytest.approx(0.0, abs=1e-8)
    if res.alpha_pre > 0:
        assert res.kappa == pytest.approx(1.0, abs=1e-8)


@pytest.mark.unit
def test_compute_erls_fixed_shape_flexible_requires_flex_mask():
    V, _e0, *_ = bardoscia_example3()
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    U_base = np.full((N, 1), 0.5, dtype=float)

    with pytest.raises(ValueError):
        compute_erls_zero_shortfall(
            V=V,
            V_tilde=V,
            U_base=U_base,
            lam=0.2,
            buffer_mode="fixed_shape_flexible",
            flex_mask=None,
        )


@pytest.mark.integration
def test_compute_erls_fixed_shape_flexible_runs_and_hits_target():
    V, _e0, *_ = bardoscia_example3()
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    U_base = np.full((N, 1), 0.5, dtype=float)
    # Make one node flexible, others fixed (arbitrary but deterministic)
    flex_mask = np.zeros((N, 1), dtype=bool)
    flex_mask[0, 0] = True

    res = compute_erls_zero_shortfall(
        V=V,
        V_tilde=V,
        U_base=U_base,
        lam=0.2,
        buffer_mode="fixed_shape_flexible",
        flex_mask=flex_mask,
        rel_tol_fpa=1e-10,
    )

    assert res.R_pre <= 1e-8
    assert res.R_post <= 1e-8
    # Identity case => no ERLS benefit expected
    assert res.erls == pytest.approx(0.0, abs=1e-8)

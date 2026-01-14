# tests/test_erls.py
from __future__ import annotations

import numpy as np
import pytest

from src.shocks import xi_from_uniform_base
from src.erls import compute_erls_zero_shortfall

from tests.fixtures import (
    bardoscia_example1,
    bardoscia_example2,
    bardoscia_example3,
)

# -------------------------
# Unit tests: xi construction
# -------------------------

@pytest.mark.unit
def test_xi_from_uniform_base_requires_correct_shape():
    V_ref = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)

    # Wrong length: N=2 but length=3
    U_wrong = np.array([0.1, 0.2, 0.3], dtype=float)
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
# Integration tests: ERLS computation (behavioural only)
# ---------------------------------

@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture_fn",
    [bardoscia_example1, bardoscia_example2, bardoscia_example3],
)
def test_compute_erls_zero_shortfall_behavioural_identity_case_gives_zero_erls(fixture_fn):
    V, _e0, *_ = fixture_fn()
    V = np.asarray(V, dtype=float)
    N = V.shape[0]

    U_base = np.full((N, 1), 0.6, dtype=float)
    lam = 0.2

    res = compute_erls_zero_shortfall(
        V=V,
        V_tilde=V,               # identity
        U_base=U_base,
        lam=lam,
        xi_scale="row_sum",
        rel_tol_fpa=1e-10,
    )

    assert res.R_pre <= 1e-8
    assert res.R_post <= 1e-8

    # identity => no ERLS benefit expected
    assert res.erls == pytest.approx(0.0, abs=1e-8)
    if res.alpha_pre > 0:
        assert res.kappa == pytest.approx(1.0, abs=1e-8)


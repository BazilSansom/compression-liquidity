import numpy as np

from src.shocks import (
    ShockShapeParams,
    UniformShockShapeParams,
    generate_gaussian_factor_shapes,
    generate_uniform_factor_shapes,
    generate_liquidity_drain_xi,
)


def empirical_mean_offdiag_corr(X: np.ndarray) -> float:
    """
    Compute mean off-diagonal pairwise correlation of columns of X.
    """
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


def test_one_sided_gaussian_shapes_are_non_negative():
    """
    One-sided Gaussian factor shapes should produce non-negative magnitudes.
    """
    N = 20
    params = ShockShapeParams(rho=0.5, one_sided=True)
    shapes = generate_gaussian_factor_shapes(N, params, n_samples=500, seed=123)

    assert np.all(shapes >= 0.0), "One-sided Gaussian shapes must be non-negative."


def test_empirical_correlation_increases_with_rho_gaussian():
    """
    For the signed Gaussian factor shapes, empirical mean pairwise correlation
    should increase with the rho parameter.
    """
    N = 30
    n_samples = 5000

    params0 = ShockShapeParams(rho=0.0, one_sided=False)
    Z0 = generate_gaussian_factor_shapes(N, params0, n_samples=n_samples, seed=1)
    mean_corr0 = empirical_mean_offdiag_corr(Z0)

    params05 = ShockShapeParams(rho=0.5, one_sided=False)
    Z05 = generate_gaussian_factor_shapes(N, params05, n_samples=n_samples, seed=2)
    mean_corr05 = empirical_mean_offdiag_corr(Z05)

    params09 = ShockShapeParams(rho=0.9, one_sided=False)
    Z09 = generate_gaussian_factor_shapes(N, params09, n_samples=n_samples, seed=3)
    mean_corr09 = empirical_mean_offdiag_corr(Z09)

    assert mean_corr0 < mean_corr05 < mean_corr09, (
        f"Empirical correlations (Gaussian) not ordered as expected:\n"
        f"rho=0.0 → {mean_corr0}\n"
        f"rho=0.5 → {mean_corr05}\n"
        f"rho=0.9 → {mean_corr09}"
    )


def test_uniform_shapes_in_unit_interval():
    """
    Uniform factor shapes should lie in (0,1).
    """
    N = 10
    params = UniformShockShapeParams(rho=0.7)
    U = generate_uniform_factor_shapes(N, params, n_samples=1000, seed=42)

    assert np.all(U > 0.0) and np.all(U < 1.0), "Uniform shapes must be in (0,1)."


def test_empirical_correlation_increases_with_rho_uniform():
    """
    For the Uniform(0,1) factor shapes obtained via the Gaussian copula, the
    empirical mean pairwise correlation should increase with the underlying
    rho parameter, even though it will not equal rho.
    """
    N = 30
    n_samples = 5000

    params0 = UniformShockShapeParams(rho=0.0)
    U0 = generate_uniform_factor_shapes(N, params0, n_samples=n_samples, seed=1)
    mean_corr0 = empirical_mean_offdiag_corr(U0)

    params05 = UniformShockShapeParams(rho=0.5)
    U05 = generate_uniform_factor_shapes(N, params05, n_samples=n_samples, seed=2)
    mean_corr05 = empirical_mean_offdiag_corr(U05)

    params09 = UniformShockShapeParams(rho=0.9)
    U09 = generate_uniform_factor_shapes(N, params09, n_samples=n_samples, seed=3)
    mean_corr09 = empirical_mean_offdiag_corr(U09)

    assert mean_corr0 < mean_corr05 < mean_corr09, (
        f"Empirical correlations (Uniform) not ordered as expected:\n"
        f"rho=0.0 → {mean_corr0}\n"
        f"rho=0.5 → {mean_corr05}\n"
        f"rho=0.9 → {mean_corr09}"
    )



def test_generate_liquidity_drain_xi_non_negative_and_shape():
    """
    xi should be N x 1 and non-negative.
    """
    V = np.array([
        [0.0, 10.0, 0.0],
        [0.0, 0.0,  5.0],
        [2.0, 0.0,  0.0],
    ])

    xi = generate_liquidity_drain_xi(V_ref=V, rho=0.5, lam=0.3, seed=123, scale="row_sum")

    assert xi.shape == (3, 1)
    assert np.all(xi >= 0.0)


def test_generate_liquidity_drain_xi_is_bounded_by_lam_times_scale():
    """
    Because U in (0,1), we should have xi_i <= lam * s_i(V_ref)
    for each scaling convention.
    """
    V = np.array([
        [0.0, 10.0, 0.0],
        [0.0, 0.0,  5.0],
        [2.0, 0.0,  0.0],
    ])
    lam = 0.7

    # row_sum scale
    xi_row = generate_liquidity_drain_xi(V_ref=V, rho=0.2, lam=lam, seed=1, scale="row_sum")
    s_row = V.sum(axis=1).reshape(-1, 1)
    assert np.all(xi_row <= lam * s_row + 1e-12)

    # col_sum scale
    xi_col = generate_liquidity_drain_xi(V_ref=V, rho=0.2, lam=lam, seed=1, scale="col_sum")
    s_col = V.sum(axis=0).reshape(-1, 1)
    assert np.all(xi_col <= lam * s_col + 1e-12)

    # total scale
    xi_tot = generate_liquidity_drain_xi(V_ref=V, rho=0.2, lam=lam, seed=1, scale="total")
    s_tot = np.full((V.shape[0], 1), V.sum() / V.shape[0], dtype=float)
    assert np.all(xi_tot <= lam * s_tot + 1e-12)


def _mean_offdiag_corr(X: np.ndarray) -> float:
    """
    Mean off-diagonal correlation of columns of X (X shape: n_samples x N).
    """
    C = np.corrcoef(X, rowvar=False)
    N = C.shape[0]
    off = C[~np.eye(N, dtype=bool)]
    return float(off.mean())


def test_generate_liquidity_drain_xi_correlation_increases_with_rho():
    """
    Empirically, dependence in xi across nodes should increase with rho,
    holding everything else fixed.

    We generate many draws by varying the seed deterministically.
    """
    N = 25

    # Use a simple reference V with equal row sums so scaling doesn't induce spurious correlation
    V = np.ones((N, N)) - np.eye(N)  # row sum = N-1
    lam = 0.5

    # Collect many samples by varying the seed
    n_samples = 400
    X_rho0 = np.vstack([
        generate_liquidity_drain_xi(V_ref=V, rho=0.0, lam=lam, seed=s, scale="row_sum").T
        for s in range(n_samples)
    ])  # shape: n_samples x N

    X_rho9 = np.vstack([
        generate_liquidity_drain_xi(V_ref=V, rho=0.9, lam=lam, seed=s, scale="row_sum").T
        for s in range(n_samples)
    ])

    corr0 = _mean_offdiag_corr(X_rho0)
    corr9 = _mean_offdiag_corr(X_rho9)

    assert corr0 < corr9, f"Expected corr(rho=0.0) < corr(rho=0.9), got {corr0} vs {corr9}"

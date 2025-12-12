import numpy as np

from src.shocks import (
    ShockShapeParams,
    UniformShockShapeParams,
    generate_gaussian_factor_shapes,
    generate_uniform_factor_shapes,
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

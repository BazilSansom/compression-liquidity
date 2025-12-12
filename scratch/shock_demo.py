"""
Plot empirical mean pairwise correlation vs. rho for the Gaussian factor
shock shape generator (signed vs one-sided).

Usage (from repo root, in .venv):

    python scratch/plot_shock_correlation.py

This will pop up a matplotlib window and also save the figure as:
    figures/shock_correlation_vs_rho.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.shocks import ShockShapeParams, generate_gaussian_factor_shapes


def empirical_mean_offdiag_corr(X: np.ndarray) -> float:
    """
    Compute mean off-diagonal pairwise correlation of columns of X.
    """
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    return float(off_diag.mean())


def main():
    # --- parameters you might tweak ---
    N = 20               # number of nodes
    n_samples = 20_000   # number of draws for each rho
    rhos = np.linspace(0.0, 0.95, 10)
    seed = 42
    # ----------------------------------

    signed_corrs = []
    one_sided_corrs = []

    for rho in rhos:
        params_signed = ShockShapeParams(rho=rho, one_sided=False)
        params_one = ShockShapeParams(rho=rho, one_sided=True)

        Z_signed = generate_gaussian_factor_shapes(
            num_nodes=N,
            params=params_signed,
            n_samples=n_samples,
            seed=seed,
        )
        Z_one = generate_gaussian_factor_shapes(
            num_nodes=N,
            params=params_one,
            n_samples=n_samples,
            seed=seed,
        )

        signed_corrs.append(empirical_mean_offdiag_corr(Z_signed))
        one_sided_corrs.append(empirical_mean_offdiag_corr(Z_one))

    # --- make plot ---
    plt.figure()
    plt.scatter(rhos, signed_corrs, label="signed", marker="o")
    plt.scatter(rhos, one_sided_corrs, label="one-sided", marker="x")
    plt.xlabel("rho (model parameter)")
    plt.ylabel("mean empirical pairwise corr")
    plt.title("Empirical correlation vs. rho in Gaussian factor model")
    plt.legend()
    plt.tight_layout()

    # ensure figures/ exists
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "shock_correlation_vs_rho.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()

"""
A synthetic generator that mimics real gene-expression statistical structure
more closely than synthetic.py's independent-Gaussian-background approach:

  - Genes within a planted bicluster share an underlying LATENT co-expression
    factor (not just a shifted mean), so the correlation structure looks like
    real co-expressed gene modules rather than a mean-shifted block.
  - Background (non-bicluster) genes also have realistic low-level pairwise
    correlation via a shared batch/library-size confound, mimicking real
    expression data where "independent" genes are rarely truly independent.
  - Noise is log-normal-flavored (heavier right tail) rather than pure
    Gaussian, closer to real expression noise.

This is still NOT real biological data -- it exists only to test whether
RCDA-MOEA's row/column diversity mechanism shows an advantage when the
correlation structure resembles real co-expression, which the simpler
generator in synthetic.py does not attempt to capture.
"""
from __future__ import annotations
import numpy as np


def generate_realistic_synthetic_matrix(
    n_genes: int = 200, n_conditions: int = 60, n_modules: int = 6,
    module_gene_range=(8, 20), module_condition_range=(6, 15),
    factor_strength: float = 2.0, background_correlation: float = 0.3,
    noise_lognormal_sigma: float = 0.4, overlap: bool = True, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, m = n_genes, n_conditions

    # Shared low-level confound (e.g. library-size-like effect per condition)
    # gives ALL genes weak background correlation, as in real expression data.
    batch_effect = rng.normal(0, 1, size=m) * np.sqrt(background_correlation)
    X = np.tile(batch_effect, (n, 1))
    X += rng.normal(0, 1, size=(n, m)) * np.sqrt(1 - background_correlation)

    for _ in range(n_modules):
        g_size = rng.integers(*module_gene_range)
        c_size = rng.integers(*module_condition_range)
        if overlap:
            r0 = rng.integers(0, max(1, n - g_size))
            c0 = rng.integers(0, max(1, m - c_size))
        else:
            r0 = rng.integers(0, max(1, n - g_size))
            c0 = rng.integers(0, max(1, m - c_size))
        # A latent co-expression factor: all genes in the module load onto the
        # SAME underlying signal across the module's conditions (correlation
        # structure), rather than all just being shifted by a constant.
        latent_factor = rng.normal(0, 1, size=c_size)
        gene_loadings = rng.normal(1.0, 0.2, size=g_size)  # slightly varying loading strength
        module_signal = factor_strength * np.outer(gene_loadings, latent_factor)
        X[r0:r0 + g_size, c0:c0 + c_size] += module_signal

    # Heavier-tailed, right-skewed noise (log-normal centered near 0 in log-space)
    # is closer to real expression noise than pure Gaussian.
    lognoise = rng.lognormal(mean=0.0, sigma=noise_lognormal_sigma, size=(n, m))
    lognoise -= lognoise.mean()
    X += lognoise
    return X


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "example_realistic.npy")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X = generate_realistic_synthetic_matrix()
    np.save(out, X)
    print(f"Saved REALISTIC synthetic dataset to {out}, shape={X.shape}")

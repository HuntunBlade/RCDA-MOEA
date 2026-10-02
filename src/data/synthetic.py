"""
Synthetic data generator with planted (ground-truth) biclusters.

This exists ONLY to let you exercise the full pipeline (algorithm, baselines,
metrics, statistical tests) before the real datasets in Table 1 (Tissp, Cocel,
Mouse, Ustilago, BCancer, GPL5175) have been sourced and verified. Numbers
produced on this synthetic data must NOT be reported as the paper's Table 2/3
results — those require the real, verified datasets.
"""
from __future__ import annotations
import numpy as np


def generate_synthetic_matrix(n: int = 60, m: int = 40, n_biclusters: int = 3,
                               bicluster_size=(10, 8), noise_std: float = 0.5,
                               seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, m))
    for k in range(n_biclusters):
        r0 = rng.integers(0, n - bicluster_size[0])
        c0 = rng.integers(0, m - bicluster_size[1])
        base = rng.normal(0, 2)
        X[r0:r0 + bicluster_size[0], c0:c0 + bicluster_size[1]] += base
    X += rng.normal(0, noise_std, size=X.shape)
    return X


if __name__ == "__main__":
    import os, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--m", type=int, default=40)
    p.add_argument("--n-biclusters", type=int, default=3)
    p.add_argument("--bicluster-rows", type=int, default=10)
    p.add_argument("--bicluster-cols", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-name", default="example.npy")
    a = p.parse_args()

    out = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", a.out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X = generate_synthetic_matrix(
        n=a.n, m=a.m, n_biclusters=a.n_biclusters,
        bicluster_size=(a.bicluster_rows, a.bicluster_cols),
        noise_std=a.noise_std, seed=a.seed,
    )
    np.save(out, X)
    print(f"Saved synthetic dataset to {out}, shape={X.shape}, noise_std={a.noise_std}")

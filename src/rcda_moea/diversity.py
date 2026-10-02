"""
Row-Column structural diversity and dimension-specific mutation allocation.
Implements Sections 4.2 and 4.3 of the paper exactly.
"""
from __future__ import annotations
import numpy as np
from .core import Individual


def _pairwise_jaccard_distance(masks: list[np.ndarray]) -> float:
    """
    Mean pairwise Jaccard distance over a population's masks (row or column).
    D = 2/(|P|(|P|-1)) * sum_{p<q} (1 - |A_p ∩ A_q| / |A_p ∪ A_q|)

    Vectorized via matrix multiplication instead of a Python-level double loop
    over pairs: intersection counts for ALL pairs come from one matrix product,
    which is orders of magnitude faster than calling numpy once per pair.
    """
    P = len(masks)
    if P < 2:
        return 0.0
    M = np.stack(masks).astype(np.float32)          # (P, L)
    inter = M @ M.T                                   # (P, P) intersection counts
    sizes = M.sum(axis=1)                             # (P,)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, 1.0)
    iu, ju = np.triu_indices(P, k=1)
    dist = 1.0 - jaccard[iu, ju]
    return float(dist.mean()) if dist.size > 0 else 0.0


def compute_row_col_diversity(pop: list[Individual]) -> tuple[float, float]:
    """Returns (D_R, D_C)."""
    D_R = _pairwise_jaccard_distance([ind.r for ind in pop])
    D_C = _pairwise_jaccard_distance([ind.c for ind in pop])
    return D_R, D_C


def compute_mutation_weights(D_R: float, D_C: float, epsilon: float,
                              sharpness: float = 1.0) -> tuple[float, float]:
    """
    a_R = (1/(D_R+eps))^sharpness, a_C = (1/(D_C+eps))^sharpness
    W_R = a_R/(a_R+a_C), W_C = a_C/(a_R+a_C)

    `sharpness` (>=0) controls how strongly a diversity imbalance is amplified
    into a mutation-allocation imbalance. sharpness=0 disables dimension-specific allocation (W_R=W_C=0.5); sharpness=1 reproduces the paper's
    original formula (Sec 4.3) exactly. Diagnostic runs on synthetic data showed that with sharpness=1 and
    epsilon=1e-2, W_R/W_C often stayed near ~0.59/0.41 even when D_R and D_C
    differed by 30-40%. This is a real non-uniform allocation and must not be
    described as equivalent to random 50/50 mutation. The empirical result is
    narrower: under the tested settings, the allocation bias was insufficient
    to produce a detectable improvement in search performance. sharpness>1
    lets a given diversity imbalance translate into a larger split.

    epsilon prevents divergence when D_R or D_C approaches zero and also pulls
    the ratio a_R/a_C=(D_C+epsilon)/(D_R+epsilon) toward 1 when the diversity
    values are on the same scale as epsilon. When D_R,D_C >> epsilon, however,
    a_R/a_C approximately equals D_C/D_R, so epsilon has little compression
    effect; allocation strength is then governed mainly by D_R/D_C and the
    sharpness exponent. This distinction is stated explicitly in the paper.
    """
    a_R = (1.0 / (D_R + epsilon)) ** sharpness
    a_C = (1.0 / (D_C + epsilon)) ** sharpness
    W_R = a_R / (a_R + a_C)
    W_C = a_C / (a_R + a_C)
    return W_R, W_C

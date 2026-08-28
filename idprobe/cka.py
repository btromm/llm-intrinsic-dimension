"""Linear CKA between layers of two models (Cheng et al. Figure H.1).

Centred Kernel Alignment (Kornblith et al. 2019) compares representations that
live in *different* feature spaces -- which is exactly the cross-model case,
where model A has d=1024 and model B has d=768. It is invariant to orthogonal
transformation and isotropic scaling, but NOT to arbitrary invertible linear
maps, which is what makes it more discriminative than plain linear regression
between representations.

Linear CKA in feature space:

    CKA(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

with X, Y column-centred. We use the feature-space form rather than the Gram form
because d << n here, so the intermediate matrices are d x d rather than n x n.

**Alignment requirement.** Row i of X and row i of Y must be the same input. We
therefore compute CKA on the *task* activations, where every model sees the same
sentence list in the same order. The ID corpora are not safe for this: they are
filtered to an exact token count, and two tokenizers keep different subsets.
"""
from __future__ import annotations

import numpy as np


def _centre(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two [n, d] representation matrices of the same n inputs."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA needs matched rows, got {X.shape[0]} vs {Y.shape[0]}")
    X, Y = _centre(X), _centre(Y)
    cross = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(cross / denom) if denom > 0 else float("nan")


def cka_matrix(load_a, n_layers_a: int, load_b, n_layers_b: int) -> np.ndarray:
    """All-pairs CKA between the layers of two models.

    `load_a(l)` / `load_b(l)` return that model's layer-l activation matrix.
    Layers are loaded once each and held, so this is O(La + Lb) reads rather
    than O(La * Lb).
    """
    A = [_centre(load_a(l)) for l in range(n_layers_a)]
    B = [_centre(load_b(l)) for l in range(n_layers_b)]
    M = np.empty((n_layers_a, n_layers_b))
    for i, x in enumerate(A):
        xx = np.linalg.norm(x.T @ x, ord="fro")
        for j, y in enumerate(B):
            yy = np.linalg.norm(y.T @ y, ord="fro")
            num = np.linalg.norm(y.T @ x, ord="fro") ** 2
            M[i, j] = num / (xx * yy) if xx * yy > 0 else np.nan
    return M


__all__ = ["linear_cka", "cka_matrix"]

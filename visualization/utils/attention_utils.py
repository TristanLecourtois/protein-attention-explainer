"""
Attention utilities for the visualization API layer.

Converts numpy arrays → JSON-serializable structures
optimised for frontend rendering performance.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def layer_to_heatmap(
    aggregated: np.ndarray,    # [L, N, N]
    layer: int,
) -> list[list[float]]:
    """
    Returns [N][N] 2D list (normalised [0,1]) for heatmap display.
    """
    A = aggregated[layer]
    lo, hi = A.min(), A.max()
    if hi > lo:
        A = (A - lo) / (hi - lo)
    return A.tolist()


def layer_to_edges(
    aggregated: np.ndarray,    # [L, N, N]
    layer: int,
    threshold: float = 0.15,
    min_separation: int = 6,
    max_edges: int = 150,
) -> list[dict]:
    """
    Returns list of {i, j, weight} for 3D line rendering.
    Only long-range pairs above threshold.
    """
    A = aggregated[layer]
    N = A.shape[0]

    edges = []
    for i in range(N):
        for j in range(i + min_separation, N):
            w = float(A[i, j])
            if w >= threshold:
                edges.append({"i": i, "j": j, "weight": w})

    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges[:max_edges]


def residue_scores_for_layer(
    residue_scores: np.ndarray,   # [L, N]
    layer: int,
) -> list[float]:
    return residue_scores[layer].tolist()


def attention_summary(aggregated: np.ndarray) -> dict:
    """
    Layer-level statistics for the UI layer profile panel.
    """
    L, N, _ = aggregated.shape
    profiles = []
    for l in range(L):
        A = aggregated[l]
        eps = 1e-10
        # Row-normalise for entropy
        A_norm = A / (A.sum(axis=-1, keepdims=True) + eps)
        entropy = float(-(A_norm * np.log(A_norm + eps)).sum(axis=-1).mean())

        # Fraction of long-range attention (sep >= 12)
        lr_mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) >= 12
        lr_ratio = float(A[lr_mask].mean() / (A.mean() + eps))

        profiles.append({
            "layer": l,
            "entropy": round(entropy, 4),
            "long_range_ratio": round(lr_ratio, 4),
            "mean": round(float(A.mean()), 6),
        })
    return {"num_layers": L, "seq_len": N, "layers": profiles}


def encode_attention_for_transfer(
    aggregated: np.ndarray,   # [L, N, N]
    layer: int,
    precision: int = 4,
) -> dict:
    """
    Encode a single layer's data efficiently for JSON transfer.
    Uses 1D flattened array + shape for minimal payload size.
    """
    A = aggregated[layer]
    N = A.shape[0]

    # Round to reduce payload size
    flat = np.round(A.flatten().astype(np.float32), precision).tolist()

    return {
        "layer": layer,
        "shape": [N, N],
        "data": flat,
    }

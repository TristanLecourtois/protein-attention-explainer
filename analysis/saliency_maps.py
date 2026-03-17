"""
Saliency / gradient-based interpretation of attention patterns.

Computes which residues are most "attended to" globally,
and identifies key long-range interactions.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class SaliencyAnalyzer:
    """
    Derives saliency-like scores from attention maps.

    Approaches:
      1. Attention rollout (Abnar & Zuidema 2020)
      2. Raw attention sum
      3. Gradient-weighted attention (if gradients available)
    """

    def attention_rollout(self, aggregated: np.ndarray) -> np.ndarray:
        """
        Attention rollout: recursively propagate attention through layers.
        aggregated: [L, N, N]
        Returns: [N, N] cumulative attention
        """
        L, N, _ = aggregated.shape

        # Add residual identity connection (0.5 self + 0.5 attention)
        eye = np.eye(N, dtype=np.float32)
        rollout = eye.copy()

        for l in range(L):
            A = 0.5 * aggregated[l] + 0.5 * eye
            # Normalise rows
            row_sums = A.sum(axis=-1, keepdims=True)
            row_sums[row_sums == 0] = 1
            A = A / row_sums
            rollout = rollout @ A

        return rollout   # [N, N]

    def residue_importance(
        self,
        aggregated: np.ndarray,    # [L, N, N]
        method: str = "rollout",
    ) -> np.ndarray:
        """
        [L, N] importance score per residue per layer.
        method: 'rollout' | 'sum' | 'max'
        """
        if method == "rollout":
            rollout = self.attention_rollout(aggregated)
            # Expand to per-layer by combining rollout with each layer
            scores = np.stack([
                (aggregated[l] * rollout).sum(axis=0)
                for l in range(aggregated.shape[0])
            ])  # [L, N]
        elif method == "sum":
            scores = aggregated.sum(axis=2)   # [L, N]
        elif method == "max":
            scores = aggregated.max(axis=2)   # [L, N]
        else:
            raise ValueError(method)

        # Normalise per layer
        maxs = scores.max(axis=1, keepdims=True)
        maxs[maxs == 0] = 1
        return scores / maxs

    def long_range_interactions(
        self,
        aggregated: np.ndarray,   # [L, N, N]
        min_separation: int = 12,
        top_k: int = 50,
    ) -> list[dict]:
        """
        Find top long-range attention edges (averaged over all layers).
        """
        mean_attn = aggregated.mean(axis=0)  # [N, N]
        N = mean_attn.shape[0]

        pairs = []
        for i in range(N):
            for j in range(i + min_separation, N):
                w = float(mean_attn[i, j])
                pairs.append({"i": i, "j": j, "weight": w, "separation": j - i})

        pairs.sort(key=lambda x: -x["weight"])
        return pairs[:top_k]

    def layer_profile(self, aggregated: np.ndarray) -> dict:
        """
        Per-layer statistics: mean attention entropy, sparsity.
        Helps identify which layers focus on local vs global structure.
        """
        L = aggregated.shape[0]
        profiles = []

        for l in range(L):
            A = aggregated[l]   # [N, N]

            # Entropy (higher = more distributed)
            eps = 1e-10
            A_norm = A / (A.sum(axis=-1, keepdims=True) + eps)
            entropy = -(A_norm * np.log(A_norm + eps)).sum(axis=-1).mean()

            # Sparsity (fraction of weights > 0.1)
            sparsity = float((A > 0.1).mean())

            # Local bias: fraction of attention within 6 residues
            N = A.shape[0]
            local_mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) <= 6
            local_ratio = float(A[local_mask].mean() / (A.mean() + eps))

            profiles.append({
                "layer": l,
                "entropy": float(entropy),
                "sparsity": sparsity,
                "local_bias": local_ratio,
            })

        return {"layers": profiles}

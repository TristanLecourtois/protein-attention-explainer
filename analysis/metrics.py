"""
Quantitative metrics for attention-structure alignment.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class AttentionMetrics:
    """
    Computes metrics linking attention patterns to structural features.
    """

    @staticmethod
    def contact_precision(
        attention_map: np.ndarray,   # [N, N] — predicted
        ca_coords: np.ndarray,       # [N, 3]
        contact_threshold: float = 8.0,
        fractions: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0),
        min_separation: int = 6,
    ) -> dict[str, float]:
        N = attention_map.shape[0]
        dist = np.sqrt(((ca_coords[:, None] - ca_coords[None, :]) ** 2).sum(-1))
        true_contacts = dist < contact_threshold

        # Long-range pairs only
        pairs = [
            (i, j, attention_map[i, j])
            for i in range(N)
            for j in range(i + min_separation, N)
        ]
        pairs.sort(key=lambda x: -x[2])

        results = {}
        for frac in fractions:
            k = max(1, int(N * frac))
            top = pairs[:k]
            hits = sum(1 for i, j, _ in top if true_contacts[i, j])
            label = f"L/{int(1/frac)}" if frac < 1 else f"{int(frac)}L"
            results[f"P@{label}"] = hits / k

        return results

    @staticmethod
    def attention_vs_distance_correlation(
        attention_map: np.ndarray,   # [N, N]
        ca_coords: np.ndarray,       # [N, 3]
        min_separation: int = 6,
    ) -> float:
        """Spearman correlation between attention weight and 1/distance."""
        from scipy.stats import spearmanr

        N = attention_map.shape[0]
        dist = np.sqrt(((ca_coords[:, None] - ca_coords[None, :]) ** 2).sum(-1))

        attn_vals, inv_dist_vals = [], []
        for i in range(N):
            for j in range(i + min_separation, N):
                attn_vals.append(attention_map[i, j])
                inv_dist_vals.append(1.0 / (dist[i, j] + 1e-8))

        corr, _ = spearmanr(attn_vals, inv_dist_vals)
        return float(corr)

    @staticmethod
    def secondary_structure_attention(
        attention_map: np.ndarray,   # [N, N]
        ss_labels: list[str],        # length N — 'H', 'E', 'C'
    ) -> dict:
        """
        Mean attention within/between secondary structure elements.
        """
        N = attention_map.shape[0]
        ss = np.array(ss_labels)

        results = {}
        for ss1 in ["H", "E", "C"]:
            for ss2 in ["H", "E", "C"]:
                mask = np.outer(ss == ss1, ss == ss2).astype(float)
                total = mask.sum()
                if total > 0:
                    results[f"{ss1}-{ss2}"] = float((attention_map * mask).sum() / total)

        return results

    @staticmethod
    def layer_interpretability_score(
        aggregated: np.ndarray,   # [L, N, N]
        ca_coords: np.ndarray,    # [N, 3]
        contact_threshold: float = 8.0,
    ) -> np.ndarray:
        """
        Per-layer contact precision@L. Useful for finding which layers
        encode structural information.
        Returns [L] float array.
        """
        L = aggregated.shape[0]
        N = ca_coords.shape[0]
        dist = np.sqrt(((ca_coords[:, None] - ca_coords[None, :]) ** 2).sum(-1))
        true_contacts = dist < contact_threshold

        scores = []
        for l in range(L):
            A = aggregated[l]
            pairs = sorted(
                [(i, j, A[i, j]) for i in range(N) for j in range(i + 6, N)],
                key=lambda x: -x[2],
            )
            k = max(1, N)
            hits = sum(1 for i, j, _ in pairs[:k] if true_contacts[i, j])
            scores.append(hits / k)

        return np.array(scores, dtype=np.float32)

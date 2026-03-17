"""
Attention extraction utilities for ESM-2 / ESMFold attention maps.

Given raw attention tensors [L, H, N, N], provides:
  - Aggregation (mean/max over heads, layer selection)
  - Symmetrisation
  - Normalisation
  - Long-range filtering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class AttentionData:
    """Container for all attention-derived information."""

    raw: np.ndarray              # [L, H, N, N]  raw per-layer-head attention
    sequence: str                # amino acid sequence (length N)
    plddt: np.ndarray            # [N]  per-residue confidence

    aggregated: np.ndarray = field(default_factory=lambda: np.array([]))   # [L, N, N]
    residue_scores: np.ndarray = field(default_factory=lambda: np.array([]))  # [N]
    contact_map: np.ndarray = field(default_factory=lambda: np.array([]))  # [N, N]

    @property
    def num_layers(self) -> int:
        return self.raw.shape[0]

    @property
    def num_heads(self) -> int:
        return self.raw.shape[1]

    @property
    def seq_len(self) -> int:
        return self.raw.shape[2]


class AttentionExtractor:
    """
    Processes raw attention maps into visualisation-ready structures.

    Pipeline:
      raw [L, H, N, N]
        → aggregate heads  → [L, N, N]
        → symmetrise        → [L, N, N]
        → normalise per layer
        → residue scores [L, N]
        → contact map [N, N]
    """

    def __init__(
        self,
        head_reduction: Literal["mean", "max", "cls"] = "mean",
        symmetrise: bool = True,
        apc_correction: bool = True,
    ):
        self.head_reduction = head_reduction
        self.symmetrise = symmetrise
        self.apc_correction = apc_correction


    def process(
        self,
        raw: np.ndarray,          # [L, H, N, N]
        sequence: str,
        plddt: Optional[np.ndarray] = None,
    ) -> AttentionData:
        if plddt is None:
            plddt = np.ones(len(sequence), dtype=np.float32)

        data = AttentionData(raw=raw.astype(np.float32), sequence=sequence, plddt=plddt)

        data.aggregated = self._aggregate_heads(raw)          # [L, N, N]
        data.aggregated = self._postprocess(data.aggregated)  # symmetrise + norm
        data.residue_scores = self._residue_scores(data.aggregated)   # [L, N]
        data.contact_map = self._contact_map(data.aggregated)         # [N, N]

        return data


    def _aggregate_heads(self, raw: np.ndarray) -> np.ndarray:
        """[L, H, N, N] → [L, N, N]"""
        if self.head_reduction == "mean":
            return raw.mean(axis=1)
        elif self.head_reduction == "max":
            return raw.max(axis=1)
        elif self.head_reduction == "cls":
            # Keep only the first head per layer (often most interpretable in ESM)
            return raw[:, 0, :, :]
        else:
            raise ValueError(self.head_reduction)


    def _postprocess(self, agg: np.ndarray) -> np.ndarray:
        """[L, N, N] → symmetrised + normalised."""
        if self.symmetrise:
            agg = 0.5 * (agg + agg.transpose(0, 2, 1))

        if self.apc_correction:
            agg = self._apply_apc(agg)

        # Min-max normalise each layer independently -> [0, 1]
        for i in range(agg.shape[0]):
            lo, hi = agg[i].min(), agg[i].max()
            if hi > lo:
                agg[i] = (agg[i] - lo) / (hi - lo)

        return agg

    @staticmethod
    def _apply_apc(agg: np.ndarray) -> np.ndarray:
        """
        Average Product Correction removes phylogenetic / background signal.
        APC(i,j) = A(i,j) - A(i,.) * A(.,j) / A(.,.)
        Applied per layer.
        """
        corrected = np.empty_like(agg)
        for l in range(agg.shape[0]):
            A = agg[l]
            mean_i = A.mean(axis=1, keepdims=True)   # [N, 1]
            mean_j = A.mean(axis=0, keepdims=True)   # [1, N]
            mean_all = A.mean()
            if mean_all > 0:
                corrected[l] = A - (mean_i * mean_j) / mean_all
            else:
                corrected[l] = A
        return np.clip(corrected, 0, None)


    def _residue_scores(self, agg: np.ndarray) -> np.ndarray:
        """
        Per-residue attention score = sum of attention it *receives*.
        Returns [L, N] normalised per layer.
        """
        scores = agg.sum(axis=1)   # [L, N]  sum over source
        # Normalise per layer
        maxs = scores.max(axis=1, keepdims=True)
        maxs[maxs == 0] = 1
        return scores / maxs

    def _contact_map(self, agg: np.ndarray) -> np.ndarray:
        """
        Contact probability: mean over last 1/3 of layers (deepest semantic layers).
        [N, N] in [0, 1].
        """
        n_layers = agg.shape[0]
        start = max(0, n_layers * 2 // 3)
        deep_layers = agg[start:]                # [L', N, N]
        contact = deep_layers.mean(axis=0)       # [N, N]
        lo, hi = contact.min(), contact.max()
        if hi > lo:
            contact = (contact - lo) / (hi - lo)
        return contact


    def get_edges(
        self,
        data: AttentionData,
        layer: int,
        threshold: float = 0.1,
        min_sequence_separation: int = 6,
        max_edges: int = 200,
    ) -> list[dict]:
        """
        Returns a list of (i, j, weight) edges for a given layer,
        filtered by threshold and sequence separation.
        """
        A = data.aggregated[layer]   # [N, N]
        N = A.shape[0]
        edges = []

        for i in range(N):
            for j in range(i + min_sequence_separation, N):
                w = float(A[i, j])
                if w >= threshold:
                    edges.append({"i": i, "j": j, "weight": w})

        # Keep top-k by weight
        edges.sort(key=lambda e: e["weight"], reverse=True)
        return edges[:max_edges]

    def get_residue_scores_for_layer(
        self,
        data: AttentionData,
        layer: int,
    ) -> list[float]:
        return data.residue_scores[layer].tolist()

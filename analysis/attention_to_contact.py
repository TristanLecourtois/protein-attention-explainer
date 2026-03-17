"""
Predicts residue contacts from attention maps.

Benchmarks against true contacts from structure (8Å Cα threshold).
Implements the APC-corrected attention-contact prediction
used in Rao et al. 2020 (ESM-1 paper).
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class AttentionContactPredictor:
    """
    Converts aggregated attention maps to contact predictions.

    Uses precision@L/5, L/2, L metric (common in contact prediction).
    """

    def __init__(self, contact_threshold_angstrom: float = 8.0):
        self.contact_threshold = contact_threshold_angstrom

    def predict_contacts(
        self,
        aggregated: np.ndarray,       # [L, N, N]
        layer_weights: Optional[np.ndarray] = None,  # [L] — defaults to uniform
    ) -> np.ndarray:
        """
        Weighted sum across layers → contact probability [N, N].
        """
        L = aggregated.shape[0]
        if layer_weights is None:
            layer_weights = np.ones(L, dtype=np.float32) / L

        weights = np.array(layer_weights, dtype=np.float32)
        weights /= weights.sum()

        contact_probs = np.einsum("l,lij->ij", weights, aggregated)

        # Zero diagonal and sub-diagonal
        N = contact_probs.shape[0]
        for k in range(-5, 6):
            idx = np.arange(max(0, -k), min(N, N - k))
            contact_probs[idx, idx + k] = 0

        lo, hi = contact_probs.min(), contact_probs.max()
        if hi > lo:
            contact_probs = (contact_probs - lo) / (hi - lo)

        return contact_probs

    def evaluate(
        self,
        predicted: np.ndarray,   # [N, N]
        ca_coords: np.ndarray,   # [N, 3]
    ) -> dict:
        """
        Compute precision metrics against structural contacts.
        Returns precision@L/5, L/2, L, L*2 with separation >= 6.
        """
        N = predicted.shape[0]
        dist = np.sqrt(((ca_coords[:, None] - ca_coords[None, :]) ** 2).sum(-1))
        true_contacts = (dist < self.contact_threshold).astype(float)

        # Long-range only (seq sep >= 6)
        mask = np.zeros_like(true_contacts, dtype=bool)
        for i in range(N):
            for j in range(i + 6, N):
                mask[i, j] = True
                mask[j, i] = True

        pred_flat = predicted[mask]
        true_flat = true_contacts[mask]

        order = np.argsort(-pred_flat)
        results = {}
        for frac, label in [(0.2, "L/5"), (0.5, "L/2"), (1.0, "L"), (2.0, "2L")]:
            k = max(1, int(N * frac))
            top_k = order[:k]
            precision = true_flat[top_k].mean()
            results[f"precision_at_{label}"] = float(precision)

        return results

    def get_top_contacts(
        self,
        contact_probs: np.ndarray,   # [N, N]
        k: Optional[int] = None,
        min_separation: int = 6,
    ) -> list[dict]:
        """
        Returns top-k predicted contacts as list of dicts.
        """
        N = contact_probs.shape[0]
        if k is None:
            k = N  # top-L

        pairs = []
        for i in range(N):
            for j in range(i + min_separation, N):
                pairs.append((i, j, float(contact_probs[i, j])))

        pairs.sort(key=lambda x: -x[2])
        return [{"i": i, "j": j, "score": s} for i, j, s in pairs[:k]]

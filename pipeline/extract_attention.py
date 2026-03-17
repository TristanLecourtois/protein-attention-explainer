"""
Standalone attention extraction from a saved job directory.

Useful for re-processing with different parameters
without re-running ESMFold.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_job(job_dir: str | Path) -> dict:
    """Load a previously computed job from disk."""
    job_dir = Path(job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    meta["pdb"] = (job_dir / "structure.pdb").read_text()
    meta["attentions_raw"] = np.load(job_dir / "attentions_raw.npy")
    meta["attentions_agg"] = np.load(job_dir / "attentions_agg.npy")
    meta["residue_scores"] = np.load(job_dir / "residue_scores.npy")
    meta["contact_map"] = np.load(job_dir / "contact_map.npy")
    meta["plddt"] = np.load(job_dir / "plddt.npy")
    return meta


def reprocess_attention(
    job_dir: str | Path,
    head_reduction: str = "mean",
    apc: bool = True,
    symmetrise: bool = True,
) -> dict:
    """Re-run attention processing with new parameters."""
    from models.attention_extractor import AttentionExtractor

    job_dir = Path(job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    raw = np.load(job_dir / "attentions_raw.npy")
    plddt = np.load(job_dir / "plddt.npy")

    extractor = AttentionExtractor(
        head_reduction=head_reduction,
        symmetrise=symmetrise,
        apc_correction=apc,
    )
    attn_data = extractor.process(raw=raw, sequence=meta["sequence"], plddt=plddt)

    return {
        "aggregated": attn_data.aggregated,
        "residue_scores": attn_data.residue_scores,
        "contact_map": attn_data.contact_map,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir")
    parser.add_argument("--head-reduction", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--no-apc", action="store_true")
    args = parser.parse_args()

    result = reprocess_attention(
        args.job_dir,
        head_reduction=args.head_reduction,
        apc=not args.no_apc,
    )
    print(f"Aggregated shape: {result['aggregated'].shape}")
    print(f"Residue scores shape: {result['residue_scores'].shape}")

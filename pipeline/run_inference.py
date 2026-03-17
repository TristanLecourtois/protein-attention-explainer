from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProteinPipeline:
    """Orchestrates ESMFold inference + attention extraction."""

    def __init__(
        self,
        results_dir: str | Path = "./data/processed",
        device: Optional[str] = None,
        use_half: bool = True,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._device = device
        self._use_half = use_half
        self._model: Optional[object] = None
        self._extractor: Optional[object] = None


    def _get_model(self):
        if self._model is None:
            from models.esmfold_wrapper import ESMFoldWrapper
            self._model = ESMFoldWrapper(
                device=self._device,
                use_half=self._use_half,
            ).load()
        return self._model

    def _get_extractor(self):
        if self._extractor is None:
            from models.attention_extractor import AttentionExtractor
            self._extractor = AttentionExtractor(
                head_reduction="mean",
                symmetrise=True,
                apc_correction=True,
            )
        return self._extractor


    def run(self, sequence: str, job_id: Optional[str] = None) -> dict:
        """
        Full pipeline.

        Returns:
          {
            "job_id": str,
            "sequence": str,
            "pdb": str,
            "plddt": list[float],
            "num_layers": int,
            "num_heads": int,
            "seq_len": int,
            "results_dir": str,   # directory where numpy files are saved
          }
        """
        t0 = time.time()
        sequence = sequence.strip().upper()

        if job_id is None:
            import hashlib
            job_id = hashlib.sha1(sequence.encode()).hexdigest()[:12]

        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # --- Check cache ---
        cached = self._load_cache(job_dir, sequence)
        if cached:
            logger.info("Cache hit for job %s", job_id)
            return cached

     
        logger.info("Starting inference for job %s (len=%d)", job_id, len(sequence))
        model = self._get_model()
        raw_result = model.predict(sequence)

      
        pdb_path = job_dir / "structure.pdb"
        pdb_path.write_text(raw_result["pdb"])

    
        extractor = self._get_extractor()
        attn_data = extractor.process(
            raw=raw_result["attentions"],
            sequence=sequence,
            plddt=raw_result["plddt"],
        )

       
        np.save(job_dir / "attentions_raw.npy", attn_data.raw)
        np.save(job_dir / "attentions_agg.npy", attn_data.aggregated)
        np.save(job_dir / "residue_scores.npy", attn_data.residue_scores)
        np.save(job_dir / "contact_map.npy", attn_data.contact_map)
        np.save(job_dir / "plddt.npy", attn_data.plddt)

     
        meta = {
            "job_id": job_id,
            "sequence": sequence,
            "seq_len": len(sequence),
            "num_layers": attn_data.num_layers,
            "num_heads": attn_data.num_heads,
            "plddt": attn_data.plddt.tolist(),
            "elapsed_seconds": round(time.time() - t0, 2),
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        logger.info("Job %s done in %.1fs", job_id, time.time() - t0)

        return {
            **meta,
            "pdb": raw_result["pdb"],
            "results_dir": str(job_dir),
        }


    def _load_cache(self, job_dir: Path, sequence: str) -> Optional[dict]:
        meta_path = job_dir / "meta.json"
        pdb_path = job_dir / "structure.pdb"
        if not (meta_path.exists() and pdb_path.exists()):
            return None

        meta = json.loads(meta_path.read_text())
        if meta.get("sequence") != sequence:
            return None  # hash collision or different sequence

        return {
            **meta,
            "pdb": pdb_path.read_text(),
            "results_dir": str(job_dir),
        }


def main():
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run ESMFold + attention extraction")
    parser.add_argument("sequence", help="Amino acid sequence (single-letter code)")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipeline = ProteinPipeline(results_dir=args.output_dir, device=args.device)
    result = pipeline.run(args.sequence)
    print(json.dumps({k: v for k, v in result.items() if k != "pdb"}, indent=2))


if __name__ == "__main__":
    main()

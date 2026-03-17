"""
API route handlers.

Job lifecycle:
  PENDING → RUNNING → DONE | ERROR
  Results cached on disk under data/processed/{job_id}/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.utils.pdb_parser import pdb_to_residue_list, pdb_with_attention_bfactors
from visualization.utils.attention_utils import (
    layer_to_edges,
    residue_scores_for_layer,
    attention_summary,
    encode_attention_for_transfer,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory job store (could be Redis in production)
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)
_websockets: dict[str, list[WebSocket]] = {}

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./data/processed"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQUENCE_LENGTH", "400"))


class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=10, max_length=1000)
    num_recycles: int = Field(1, ge=1, le=4)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(v) - valid
        if invalid:
            raise ValueError(f"Invalid amino acid characters: {invalid}")
        if len(v) > MAX_SEQ_LEN:
            raise ValueError(f"Sequence too long (max {MAX_SEQ_LEN})")
        return v


class JobStatus(BaseModel):
    job_id: str
    status: str        
    progress: int      
    message: str
    meta: Optional[dict] = None


def _load_job_arrays(job_id: str) -> dict:
    """Load numpy arrays for a completed job."""
    job_dir = RESULTS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, f"Job {job_id} not found on disk")

    return {
        "aggregated": np.load(job_dir / "attentions_agg.npy"),
        "residue_scores": np.load(job_dir / "residue_scores.npy"),
        "contact_map": np.load(job_dir / "contact_map.npy"),
        "plddt": np.load(job_dir / "plddt.npy"),
        "pdb": (job_dir / "structure.pdb").read_text(),
        "meta": json.loads((job_dir / "meta.json").read_text()),
    }


def _ensure_done(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        # Try to load from disk (server restart)
        job_dir = RESULTS_DIR / job_id
        if (job_dir / "meta.json").exists():
            meta = json.loads((job_dir / "meta.json").read_text())
            _jobs[job_id] = {"status": "DONE", "progress": 100, "message": "Loaded from cache", "meta": meta}
            return _jobs[job_id]
        raise HTTPException(404, f"Job {job_id} not found")

    if job["status"] == "ERROR":
        raise HTTPException(500, job.get("message", "Job failed"))
    if job["status"] != "DONE":
        raise HTTPException(202, f"Job not ready: {job['status']}")

    return job


async def _notify_ws(job_id: str, message: dict):
    """Broadcast a message to all WebSocket clients watching this job."""
    clients = _websockets.get(job_id, [])
    dead = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


def _run_pipeline(job_id: str, sequence: str, num_recycles: int):
    """Runs in a thread pool — synchronous."""
    try:
        _jobs[job_id]["status"] = "RUNNING"
        _jobs[job_id]["progress"] = 5
        _jobs[job_id]["message"] = "Loading model…"

        from pipeline.run_inference import ProteinPipeline

        pipeline = ProteinPipeline(results_dir=RESULTS_DIR)

        _jobs[job_id]["progress"] = 20
        _jobs[job_id]["message"] = "Running ESMFold…"

        result = pipeline.run(sequence, job_id=job_id)

        _jobs[job_id]["status"] = "DONE"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["message"] = f"Done in {result.get('elapsed_seconds', '?')}s"
        _jobs[job_id]["meta"] = {
            k: v for k, v in result.items() if k not in ("pdb",)
        }

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        _jobs[job_id]["status"] = "ERROR"
        _jobs[job_id]["message"] = str(exc)


@router.post("/predict", response_model=JobStatus, status_code=202)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """Submit a sequence for structure prediction + attention extraction."""
    import hashlib
    job_id = hashlib.sha1(req.sequence.encode()).hexdigest()[:12]

    if job_id not in _jobs:
        _jobs[job_id] = {
            "status": "PENDING",
            "progress": 0,
            "message": "Queued",
            "meta": None,
        }
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            _executor,
            _run_pipeline,
            job_id,
            req.sequence,
            req.num_recycles,
        )

    return JobStatus(job_id=job_id, **_jobs[job_id])


@router.get("/jobs/{job_id}/status", response_model=JobStatus)
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        # Check disk cache
        job_dir = RESULTS_DIR / job_id
        if (job_dir / "meta.json").exists():
            meta = json.loads((job_dir / "meta.json").read_text())
            return JobStatus(job_id=job_id, status="DONE", progress=100,
                             message="Loaded from cache", meta=meta)
        raise HTTPException(404, "Job not found")
    return JobStatus(job_id=job_id, **job)


@router.get("/jobs/{job_id}/pdb")
async def get_pdb(job_id: str, attention_layer: Optional[int] = None):
    """
    Returns the PDB string.
    If attention_layer is provided, B-factors are replaced with attention scores.
    """
    _ensure_done(job_id)
    arrays = _load_job_arrays(job_id)
    pdb = arrays["pdb"]

    if attention_layer is not None:
        n_layers = arrays["aggregated"].shape[0]
        layer = max(0, min(attention_layer, n_layers - 1))
        scores = residue_scores_for_layer(arrays["residue_scores"], layer)
        pdb = pdb_with_attention_bfactors(pdb, scores)

    return {"pdb": pdb}


@router.get("/jobs/{job_id}/residues")
async def get_residues(job_id: str):
    """Residue list with CA coordinates + pLDDT."""
    _ensure_done(job_id)
    arrays = _load_job_arrays(job_id)
    residues = pdb_to_residue_list(arrays["pdb"])

    # Attach pLDDT
    plddt = arrays["plddt"].tolist()
    for i, r in enumerate(residues):
        if i < len(plddt):
            r["plddt"] = round(plddt[i], 2)

    return {"residues": residues}


@router.get("/jobs/{job_id}/attention")
async def get_attention(
    job_id: str,
    layer: int = 0,
    threshold: float = 0.15,
    min_separation: int = 6,
    max_edges: int = 150,
):
    """
    Returns attention data for one layer:
    - residue_scores: [N] float
    - edges: [{i, j, weight}] above threshold
    - heatmap_flat: flattened [N*N] for matrix view
    """
    _ensure_done(job_id)
    arrays = _load_job_arrays(job_id)
    n_layers = arrays["aggregated"].shape[0]
    layer = max(0, min(layer, n_layers - 1))

    scores = residue_scores_for_layer(arrays["residue_scores"], layer)
    edges = layer_to_edges(
        arrays["aggregated"], layer,
        threshold=threshold,
        min_separation=min_separation,
        max_edges=max_edges,
    )
    encoded = encode_attention_for_transfer(arrays["aggregated"], layer)

    return {
        "layer": layer,
        "residue_scores": scores,
        "edges": edges,
        "heatmap": encoded,
        "contact_map": arrays["contact_map"].tolist(),
    }


@router.get("/jobs/{job_id}/summary")
async def get_summary(job_id: str):
    """Layer-level statistics + metadata."""
    _ensure_done(job_id)
    arrays = _load_job_arrays(job_id)
    summary = attention_summary(arrays["aggregated"])
    summary["meta"] = arrays["meta"]
    return summary


@router.get("/jobs/{job_id}/layer-profile")
async def get_layer_profile(job_id: str):
    """Per-layer interpretability scores (contact precision@L)."""
    from analysis.saliency_maps import SaliencyAnalyzer

    _ensure_done(job_id)
    arrays = _load_job_arrays(job_id)
    analyzer = SaliencyAnalyzer()
    profile = analyzer.layer_profile(arrays["aggregated"])
    return profile


@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in _websockets:
        _websockets[job_id] = []
    _websockets[job_id].append(websocket)

    try:
        while True:
            job = _jobs.get(job_id, {})
            if job:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.get("status", "UNKNOWN"),
                    "progress": job.get("progress", 0),
                    "message": job.get("message", ""),
                })
            if job.get("status") in ("DONE", "ERROR"):
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _websockets:
            _websockets[job_id] = [w for w in _websockets[job_id] if w != websocket]

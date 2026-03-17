"""
FastAPI application — serves the protein attention explainer API.

Endpoints:
  POST /api/predict          — submit sequence, get job_id
  GET  /api/jobs/{id}/status — polling
  GET  /api/jobs/{id}/pdb    — PDB string (optionally with attention bfactors)
  GET  /api/jobs/{id}/attention — attention data for a given layer
  GET  /api/jobs/{id}/residues  — residue list with CA coords
  GET  /api/jobs/{id}/edges     — attention edges for 3D lines
  GET  /api/jobs/{id}/summary   — layer-level statistics
  WS   /ws/{id}              — real-time progress updates
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Protein Attention Explainer API")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Protein Attention Explainer",
    description="ESMFold + attention map visualization API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
from visualization.backend.routes import router  # noqa: E402
app.include_router(router, prefix="/api")

# Serve frontend build in production
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "visualization.backend.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )

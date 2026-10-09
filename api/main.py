"""FastAPI service exposing the core detection and counting module."""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from geoforest import __version__
from geoforest.detector import get_detector
from geoforest.pipeline import run_pipeline

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_EDGE_PX = 4096

app = FastAPI(
    title="GeoForest AI",
    version=__version__,
    summary="Individual tree detection and counting from satellite imagery.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_detector = None


def detector():
    """Load the detector once, on first request rather than at import."""
    global _detector
    if _detector is None:
        _detector = get_detector()
        logger.info("detector ready: %s", _detector.name)
    return _detector


class Health(BaseModel):
    status: str
    version: str
    detector: str
    is_real_model: bool


class DetectionRecord(BaseModel):
    id: int
    x0: float
    y0: float
    x1: float
    y1: float
    score: float


class DetectResponse(BaseModel):
    count: int
    raw_count: int
    duplicates_removed: int
    tile_count: int
    elapsed_s: float
    detector_name: str
    is_real_model: bool
    tile_size: int
    overlap: float
    iou_threshold: float
    scene_width: int
    scene_height: int
    detections: list[DetectionRecord]


@app.get("/health", response_model=Health)
def health() -> Health:
    active = detector()
    return Health(
        status="ok",
        version=__version__,
        detector=active.name,
        is_real_model=active.is_real_model,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    tile_size: int = 400,
    overlap: float = 0.25,
    iou_threshold: float = 0.3,
) -> DetectResponse:
    """Detect and count tree crowns in an uploaded scene."""
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415, detail="file is not a readable image"
        ) from exc

    if max(image.size) > MAX_EDGE_PX:
        raise HTTPException(
            status_code=413,
            detail=f"image edge exceeds {MAX_EDGE_PX} px; tile it before upload",
        )

    try:
        result = run_pipeline(
            np.asarray(image),
            detector=detector(),
            tile_size=tile_size,
            overlap=overlap,
            iou_threshold=iou_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DetectResponse(**result.to_summary(), detections=result.to_records())

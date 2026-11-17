"""Video ingestion and validation API routes."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel

from app.config import settings
from app.video.validator import validate_video_file, is_supported_video_format, SUPPORTED_VIDEO_EXTENSIONS

router = APIRouter(prefix="/api/v1/video", tags=["Video"])


def get_sample_dir() -> Path:
    return settings.sample_videos_dir


def get_upload_dir() -> Path:
    d = settings.sample_videos_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


class VideoMetadataResponse(BaseModel):
    file_path: str
    is_valid: bool
    format: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    message: str


class SampleVideoItem(BaseModel):
    filename: str
    file_path: str
    size_bytes: int


@router.post("/upload", response_model=VideoMetadataResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(file: UploadFile = File(...)):
    """Upload and validate drone video footage for processing."""
    if not is_supported_video_format(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed extensions: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
        )

    upload_dir = get_upload_dir()
    dest_path = upload_dir / file.filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_valid, error_message, info = validate_video_file(str(dest_path))
    if not is_valid or not info:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid video: {error_message}")

    return VideoMetadataResponse(
        file_path=str(dest_path),
        is_valid=True,
        format=info.codec or "unknown",
        width=info.width,
        height=info.height,
        fps=info.fps,
        frame_count=info.frame_count,
        duration_seconds=info.duration,
        message="Video uploaded and validated successfully",
    )


@router.get("/info", response_model=VideoMetadataResponse)
async def get_video_info(path: str):
    """Retrieve metadata for an existing video file."""
    # Check direct path or path relative to base_dir
    resolved_path = path
    if not os.path.exists(resolved_path):
        candidate = str(settings.base_dir / path)
        if os.path.exists(candidate):
            resolved_path = candidate
        else:
            raise HTTPException(status_code=404, detail=f"Video file not found at '{path}'")

    is_valid, error_message, info = validate_video_file(resolved_path)
    if not is_valid or not info:
        raise HTTPException(status_code=400, detail=f"Invalid video file: {error_message}")

    return VideoMetadataResponse(
        file_path=resolved_path,
        is_valid=True,
        format=info.codec or "unknown",
        width=info.width,
        height=info.height,
        fps=info.fps,
        frame_count=info.frame_count,
        duration_seconds=info.duration,
        message="Video info extracted successfully",
    )


@router.get("/samples", response_model=List[SampleVideoItem])
async def list_sample_videos():
    """List available sample drone video recordings."""
    sample_dir = get_sample_dir()
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for f in sample_dir.rglob("*"):
        if f.is_file() and is_supported_video_format(f.name):
            samples.append(SampleVideoItem(
                filename=f.name,
                file_path=str(f),
                size_bytes=f.stat().st_size,
            ))
    return samples

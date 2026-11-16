from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Text,
    DateTime,
    Integer,
    Float,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database.base import Base


class DetectionRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("drone.id"), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    num_detections: Mapped[int] = mapped_column(Integer, default=0)
    detections_data: Mapped[list] = mapped_column(JSON, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str] = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    drone: Mapped["Drone"] = relationship(back_populates="detections")

    __table_args__ = (
        Index("ix_detection_drone_time", "drone_id", "timestamp"),
        Index("ix_detection_frame", "drone_id", "frame_number"),
    )


class TrackRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("drone.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, default=0)
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    first_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bbox_history: Mapped[list] = mapped_column(JSON, nullable=False)
    centroid_history: Mapped[list] = mapped_column(JSON, nullable=False)
    zone_transitions: Mapped[list] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_track_drone", "drone_id"),
        Index("ix_track_active", "drone_id", "is_active"),
    )


class ReIDRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    drone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("drone.id"), nullable=False)
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_reid_track", "drone_id", "track_id"),
        Index("ix_reid_time", "drone_id", "timestamp"),
    )
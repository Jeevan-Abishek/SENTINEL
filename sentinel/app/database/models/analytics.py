from datetime import datetime
from typing import Optional
from sqlalchemy import (
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


class AnalyticsRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zone.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    current_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_tracks: Mapped[int] = mapped_column(Integer, default=0)
    entries: Mapped[int] = mapped_column(Integer, default=0)
    exits: Mapped[int] = mapped_column(Integer, default=0)
    density_per_sqm: Mapped[float] = mapped_column(Float, default=0.0)
    occupancy_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    avg_speed: Mapped[float] = mapped_column(Float, default=0.0)
    flow_rate: Mapped[float] = mapped_column(Float, default=0.0)
    movement_direction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bottleneck_score: Mapped[float] = mapped_column(Float, default=0.0)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    zone: Mapped["Zone"] = relationship(back_populates="analytics")

    __table_args__ = (
        Index("ix_analytics_zone_time", "zone_id", "timestamp"),
    )


class DensityHeatmapRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zone.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    grid_data: Mapped[list] = mapped_column(JSON, nullable=False)
    grid_width: Mapped[int] = mapped_column(Integer, nullable=False)
    grid_height: Mapped[int] = mapped_column(Integer, nullable=False)
    cell_size_meters: Mapped[float] = mapped_column(Float, nullable=False)
    max_density: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_heatmap_zone_time", "zone_id", "timestamp"),
    )
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String,
    Text,
    DateTime,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    Enum as SQLEnum,
    Index,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.database.base import Base


class EventStatus(str, enum.Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Event(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location_name: Mapped[str] = mapped_column(String(255), nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[EventStatus] = mapped_column(
        SQLEnum(EventStatus), default=EventStatus.PLANNED, nullable=False
    )
    boundary_geojson: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    zones: Mapped[list["Zone"]] = relationship(back_populates="event", cascade="all, delete-orphan")
    drones: Mapped[list["Drone"]] = relationship(back_populates="event", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_event_status", "status"),
        Index("ix_event_time", "start_time", "end_time"),
    )


class ZoneType(str, enum.Enum):
    MONITORING = "monitoring"
    ENTRY = "entry"
    EXIT = "exit"
    EMERGENCY_EXIT = "emergency_exit"
    RESTRICTED = "restricted"


class Zone(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("event.id"), nullable=False)
    zone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_type: Mapped[ZoneType] = mapped_column(SQLEnum(ZoneType), default=ZoneType.MONITORING, nullable=False)
    boundary_geojson: Mapped[dict] = mapped_column(JSON, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, default=0)
    warning_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    critical_threshold: Mapped[float] = mapped_column(Float, default=0.9)
    entry_line_geojson: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    exit_line_geojson: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    calibration_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    event: Mapped["Event"] = relationship(back_populates="zones")
    analytics: Mapped[list["AnalyticsRecord"]] = relationship(back_populates="zone", cascade="all, delete-orphan")
    risk_records: Mapped[list["RiskRecord"]] = relationship(back_populates="zone", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_zone_event", "event_id"),
        Index("ix_zone_type", "zone_type"),
    )


class DroneStatus(str, enum.Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    STREAMING = "streaming"
    ERROR = "error"


class Drone(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("event.id"), nullable=False)
    drone_id: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stream_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[DroneStatus] = mapped_column(SQLEnum(DroneStatus), default=DroneStatus.OFFLINE, nullable=False)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    telemetry_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    calibration_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    event: Mapped["Event"] = relationship(back_populates="drones")
    detections: Mapped[list["DetectionRecord"]] = relationship(back_populates="drone", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_drone_event", "event_id"),
        Index("ix_drone_status", "status"),
    )
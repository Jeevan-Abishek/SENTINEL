from datetime import datetime
from typing import Optional
from sqlalchemy import (
    DateTime,
    Integer,
    Float,
    String,
    ForeignKey,
    Index,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.database.base import Base


class RiskLevel(str, enum.Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskRecord(Base):
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zone.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[RiskLevel] = mapped_column(SQLEnum(RiskLevel), default=RiskLevel.SAFE, nullable=False)
    density_score: Mapped[float] = mapped_column(Float, default=0.0)
    occupancy_score: Mapped[float] = mapped_column(Float, default=0.0)
    movement_score: Mapped[float] = mapped_column(Float, default=0.0)
    flow_score: Mapped[float] = mapped_column(Float, default=0.0)
    bottleneck_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_factors: Mapped[dict] = mapped_column(JSON, nullable=False)
    raw_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    zone: Mapped["Zone"] = relationship(back_populates="risk_records")

    __table_args__ = (
        Index("ix_risk_zone_time", "zone_id", "timestamp"),
        Index("ix_risk_level", "risk_level"),
    )
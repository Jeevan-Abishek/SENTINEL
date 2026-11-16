from typing import Optional, List
from pydantic import BaseModel, Field


class ZoneOccupancyConfig(BaseModel):
    zone_id: str
    name: str
    zone_type: str = "monitoring"
    capacity: int = Field(default=0, ge=0)
    warning_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    critical_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    entry_line: Optional[List[List[float]]] = None
    exit_line: Optional[List[List[float]]] = None
    polygon: Optional[List[List[float]]] = None
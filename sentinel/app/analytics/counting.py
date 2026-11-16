from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    CountingResult, 
    ZoneCounter, 
    AnalyticsContext
)
from app.tracking.models import Track


class PeopleCounter:
    def __init__(self, zone_id: str, zone_polygon: Optional[List[List[float]]] = None):
        self.counter = ZoneCounter(zone_id, zone_polygon)
        self.zone_id = zone_id

    def update(self, tracks: List[Track], context: AnalyticsContext, frame_rate: float = 30.0) -> CountingResult:
        self.counter.update(tracks, context.frame_number, frame_rate)
        return self.counter.get_counting_result(context.timestamp)

    def get_current_count(self) -> int:
        return self.counter.current_count

    def get_total_entries(self) -> int:
        return self.counter.entries

    def get_total_exits(self) -> int:
        return self.counter.exits

    def get_unique_tracks(self) -> int:
        return len(self.counter.unique_tracks)

    def reset(self):
        self.counter = ZoneCounter(self.zone_id, self.counter.zone_polygon.tolist() if self.counter.zone_polygon is not None else None)
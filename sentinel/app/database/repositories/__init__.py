from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.database.models import (
    Event, Zone, Drone, DetectionRecord, TrackRecord,
    AnalyticsRecord, DensityHeatmapRecord, RiskRecord, Alert, Operator,
    EventStatus, ZoneType, DroneStatus, RiskLevel, AlertStatus, AlertSeverity
)
from app.database.base import async_session_maker


class BaseRepository:
    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session
        self._own_session = session is None
        self._ctx = None
        self._nesting_depth = 0

    async def __aenter__(self):
        if self._own_session:
            if self._nesting_depth == 0:
                self._ctx = async_session_maker()
                self._session = await self._ctx.__aenter__()
            self._nesting_depth += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session:
            self._nesting_depth -= 1
            if self._nesting_depth == 0 and self._ctx:
                try:
                    if exc_type is not None:
                        await self._session.rollback()
                    else:
                        await self._session.commit()
                finally:
                    await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
                    self._session = None
                    self._ctx = None


class EventRepository(BaseRepository):
    async def create(
        self,
        name: str,
        location: str,
        latitude: float = 0.0,
        longitude: float = 0.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        status: str = "planned",
    ) -> Event:
        async with self:
            ev_status = EventStatus(status) if isinstance(status, str) else status
            event = Event(
                name=name,
                location_name=location,
                latitude=latitude,
                longitude=longitude,
                start_time=start_time or datetime.now(),
                end_time=end_time,
                description=description,
                status=ev_status,
            )
            self._session.add(event)
            await self._session.flush()
            return event

    async def get_by_id(self, event_id: uuid.UUID) -> Optional[Event]:
        async with self:
            result = await self._session.execute(select(Event).where(Event.id == event_id))
            return result.scalar_one_or_none()

    async def list_all(self, status: Optional[str] = None) -> List[Event]:
        async with self:
            query = select(Event)
            if status:
                query = query.where(Event.status == EventStatus(status))
            result = await self._session.execute(query.order_by(Event.created_at.desc()))
            return list(result.scalars().all())

    async def update(
        self,
        event_id: uuid.UUID,
        name: Optional[str] = None,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[Event]:
        async with self:
            event = await self.get_by_id(event_id)
            if not event:
                return None
            if name is not None:
                event.name = name
            if location is not None:
                event.location_name = location
            if latitude is not None:
                event.latitude = latitude
            if longitude is not None:
                event.longitude = longitude
            if start_time is not None:
                event.start_time = start_time
            if end_time is not None:
                event.end_time = end_time
            if description is not None:
                event.description = description
            if status is not None:
                event.status = EventStatus(status)
            await self._session.flush()
            return event

    async def delete(self, event_id: uuid.UUID) -> bool:
        async with self:
            event = await self.get_by_id(event_id)
            if not event:
                return False
            await self._session.delete(event)
            return True


class ZoneRepository(BaseRepository):
    async def create(
        self,
        event_id: uuid.UUID,
        name: str,
        zone_type: str = "monitoring",
        polygon: Optional[List[List[float]]] = None,
        capacity: int = 0,
        zone_id_str: Optional[str] = None,
    ) -> Zone:
        async with self:
            coords = [polygon] if polygon else [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]]
            boundary_geojson = {
                "type": "Polygon",
                "coordinates": coords
            }
            z_type = ZoneType(zone_type) if isinstance(zone_type, str) else zone_type
            zone = Zone(
                event_id=event_id,
                zone_id=zone_id_str or f"{z_type.value}_{uuid.uuid4().hex[:8]}",
                name=name,
                zone_type=z_type,
                boundary_geojson=boundary_geojson,
                capacity=capacity,
            )
            self._session.add(zone)
            await self._session.flush()
            return zone

    async def get_by_id(self, zone_id: uuid.UUID) -> Optional[Zone]:
        async with self:
            result = await self._session.execute(select(Zone).where(Zone.id == zone_id))
            return result.scalar_one_or_none()

    async def get_by_event(self, event_id: uuid.UUID) -> List[Zone]:
        async with self:
            result = await self._session.execute(select(Zone).where(Zone.event_id == event_id))
            return list(result.scalars().all())

    async def list_all(self) -> List[Zone]:
        async with self:
            result = await self._session.execute(select(Zone))
            return list(result.scalars().all())

    async def update(
        self,
        zone_id: uuid.UUID,
        name: Optional[str] = None,
        zone_type: Optional[str] = None,
        polygon: Optional[List[List[float]]] = None,
        capacity: Optional[int] = None,
    ) -> Optional[Zone]:
        async with self:
            zone = await self.get_by_id(zone_id)
            if not zone:
                return None
            if name is not None:
                zone.name = name
            if zone_type is not None:
                zone.zone_type = ZoneType(zone_type)
            if polygon is not None:
                zone.boundary_geojson = {"type": "Polygon", "coordinates": [polygon]}
            if capacity is not None:
                zone.capacity = capacity
            await self._session.flush()
            return zone

    async def delete(self, zone_id: uuid.UUID) -> bool:
        async with self:
            zone = await self.get_by_id(zone_id)
            if not zone:
                return False
            await self._session.delete(zone)
            return True


class DroneRepository(BaseRepository):
    async def create(
        self,
        event_id: uuid.UUID,
        name: str,
        rtsp_url: Optional[str] = None,
        model: Optional[str] = None,
        status: str = "offline",
    ) -> Drone:
        async with self:
            d_status = DroneStatus(status) if isinstance(status, str) else status
            drone = Drone(
                event_id=event_id,
                drone_id=name.lower().replace(" ", "-") + "-" + str(uuid.uuid4())[:8],
                name=name,
                model=model,
                stream_url=rtsp_url,
                status=d_status,
            )
            self._session.add(drone)
            await self._session.flush()
            return drone

    async def get_by_id(self, drone_id: uuid.UUID) -> Optional[Drone]:
        async with self:
            result = await self._session.execute(select(Drone).where(Drone.id == drone_id))
            return result.scalar_one_or_none()

    async def get_by_event(self, event_id: uuid.UUID) -> List[Drone]:
        async with self:
            result = await self._session.execute(select(Drone).where(Drone.event_id == event_id))
            return list(result.scalars().all())

    async def list_all(self) -> List[Drone]:
        async with self:
            result = await self._session.execute(select(Drone))
            return list(result.scalars().all())

    async def update(
        self,
        drone_id: uuid.UUID,
        name: Optional[str] = None,
        rtsp_url: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[Drone]:
        async with self:
            drone = await self.get_by_id(drone_id)
            if not drone:
                return None
            if name is not None:
                drone.name = name
            if rtsp_url is not None:
                drone.stream_url = rtsp_url
            if status is not None:
                drone.status = DroneStatus(status)
            await self._session.flush()
            return drone

    async def delete(self, drone_id: uuid.UUID) -> bool:
        async with self:
            drone = await self.get_by_id(drone_id)
            if not drone:
                return False
            await self._session.delete(drone)
            return True


class DetectionRepository(BaseRepository):
    async def create(
        self,
        drone_id: uuid.UUID,
        frame_number: int,
        timestamp: datetime,
        num_detections: int,
        detections_data: List[dict],
        processing_time_ms: float,
        model_version: str,
    ) -> DetectionRecord:
        async with self:
            detection = DetectionRecord(
                drone_id=drone_id,
                frame_number=frame_number,
                timestamp=timestamp,
                num_detections=num_detections,
                detections_data=detections_data,
                processing_time_ms=processing_time_ms,
                model_version=model_version,
            )
            self._session.add(detection)
            await self._session.flush()
            return detection

    async def get_by_id(self, detection_id: uuid.UUID) -> Optional[DetectionRecord]:
        async with self:
            result = await self._session.execute(select(DetectionRecord).where(DetectionRecord.id == detection_id))
            return result.scalar_one_or_none()

    async def get_by_drone(
        self,
        drone_id: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DetectionRecord]:
        async with self:
            query = select(DetectionRecord).where(DetectionRecord.drone_id == drone_id)
            if start_time:
                query = query.where(DetectionRecord.timestamp >= start_time)
            if end_time:
                query = query.where(DetectionRecord.timestamp <= end_time)
            query = query.order_by(DetectionRecord.timestamp.desc()).limit(limit)
            result = await self._session.execute(query)
            return list(result.scalars().all())


class TrackRepository(BaseRepository):
    async def create(
        self,
        drone_id: uuid.UUID,
        track_id: int,
        class_id: int,
        first_frame: int,
        last_frame: int,
        first_timestamp: datetime,
        last_timestamp: datetime,
        bbox_history: List[List[float]],
        centroid_history: List[List[float]],
        zone_transitions: Optional[List[dict]] = None,
    ) -> TrackRecord:
        async with self:
            track = TrackRecord(
                drone_id=drone_id,
                track_id=track_id,
                class_id=class_id,
                first_frame=first_frame,
                last_frame=last_frame,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                bbox_history=bbox_history,
                centroid_history=centroid_history,
                zone_transitions=zone_transitions,
            )
            self._session.add(track)
            await self._session.flush()
            return track

    async def get_by_id(self, track_id: uuid.UUID) -> Optional[TrackRecord]:
        async with self:
            result = await self._session.execute(select(TrackRecord).where(TrackRecord.id == track_id))
            return result.scalar_one_or_none()

    async def list_recent(self, active_only: bool = True, limit: int = 100) -> List[TrackRecord]:
        async with self:
            query = select(TrackRecord)
            if active_only:
                query = query.where(TrackRecord.is_active == True)
            query = query.order_by(TrackRecord.updated_at.desc()).limit(limit)
            result = await self._session.execute(query)
            return list(result.scalars().all())


class AnalyticsRepository(BaseRepository):
    async def create(
        self,
        zone_id: uuid.UUID,
        timestamp: datetime,
        current_count: int,
        unique_tracks: int,
        entries: int,
        exits: int,
        density_per_sqm: float,
        occupancy_ratio: float,
        avg_speed: float,
        flow_rate: float,
        movement_direction: Optional[float] = None,
        bottleneck_score: float = 0.0,
        raw_data: Optional[dict] = None,
    ) -> AnalyticsRecord:
        async with self:
            analytics = AnalyticsRecord(
                zone_id=zone_id,
                timestamp=timestamp,
                current_count=current_count,
                unique_tracks=unique_tracks,
                entries=entries,
                exits=exits,
                density_per_sqm=density_per_sqm,
                occupancy_ratio=occupancy_ratio,
                avg_speed=avg_speed,
                flow_rate=flow_rate,
                movement_direction=movement_direction,
                bottleneck_score=bottleneck_score,
                raw_data=raw_data,
            )
            self._session.add(analytics)
            await self._session.flush()
            return analytics

    async def get_by_id(self, analytics_id: uuid.UUID) -> Optional[AnalyticsRecord]:
        async with self:
            result = await self._session.execute(select(AnalyticsRecord).where(AnalyticsRecord.id == analytics_id))
            return result.scalar_one_or_none()

    async def get_by_zone(
        self,
        zone_id: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AnalyticsRecord]:
        async with self:
            query = select(AnalyticsRecord).where(AnalyticsRecord.zone_id == zone_id)
            if start_time:
                query = query.where(AnalyticsRecord.timestamp >= start_time)
            if end_time:
                query = query.where(AnalyticsRecord.timestamp <= end_time)
            query = query.order_by(AnalyticsRecord.timestamp.desc()).limit(limit)
            result = await self._session.execute(query)
            return list(result.scalars().all())


class RiskRepository(BaseRepository):
    async def create(
        self,
        zone_id: uuid.UUID,
        timestamp: datetime,
        risk_level: str,
        risk_score: float,
        risk_factors: Optional[dict] = None,
        density_score: float = 0.0,
        occupancy_score: float = 0.0,
        movement_score: float = 0.0,
        flow_score: float = 0.0,
        bottleneck_score: float = 0.0,
        raw_data: Optional[dict] = None,
    ) -> RiskRecord:
        async with self:
            level = RiskLevel(risk_level) if isinstance(risk_level, str) else risk_level
            risk = RiskRecord(
                zone_id=zone_id,
                timestamp=timestamp,
                risk_level=level,
                risk_score=risk_score,
                density_score=density_score,
                occupancy_score=occupancy_score,
                movement_score=movement_score,
                flow_score=flow_score,
                bottleneck_score=bottleneck_score,
                risk_factors=risk_factors or {},
                raw_data=raw_data or {},
            )
            self._session.add(risk)
            await self._session.flush()
            return risk

    async def get_by_id(self, risk_id: uuid.UUID) -> Optional[RiskRecord]:
        async with self:
            result = await self._session.execute(select(RiskRecord).where(RiskRecord.id == risk_id))
            return result.scalar_one_or_none()

    async def get_by_zone(
        self,
        zone_id: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RiskRecord]:
        async with self:
            query = select(RiskRecord).where(RiskRecord.zone_id == zone_id)
            if start_time:
                query = query.where(RiskRecord.timestamp >= start_time)
            if end_time:
                query = query.where(RiskRecord.timestamp <= end_time)
            query = query.order_by(RiskRecord.timestamp.desc()).limit(limit)
            result = await self._session.execute(query)
            return list(result.scalars().all())


class AlertRepository(BaseRepository):
    async def create(
        self,
        event_id: uuid.UUID,
        title: str,
        message: str,
        reason: str,
        risk_level: str = "warning",
        zone_id: Optional[uuid.UUID] = None,
        drone_id: Optional[uuid.UUID] = None,
        timestamp: Optional[datetime] = None,
        risk_score: float = 0.0,
        alert_metadata: Optional[dict] = None,
    ) -> Alert:
        async with self:
            severity = AlertSeverity(risk_level) if isinstance(risk_level, str) else risk_level
            alert = Alert(
                event_id=event_id,
                zone_id=zone_id,
                drone_id=drone_id,
                title=title,
                message=message,
                reason=reason,
                risk_level=severity,
                risk_score=risk_score,
                timestamp=timestamp or datetime.now(),
                status=AlertStatus.ACTIVE,
                alert_metadata=alert_metadata or {},
            )
            self._session.add(alert)
            await self._session.flush()
            return alert

    async def get_by_id(self, alert_id: uuid.UUID) -> Optional[Alert]:
        async with self:
            result = await self._session.execute(select(Alert).where(Alert.id == alert_id))
            return result.scalar_one_or_none()

    async def list_filtered(
        self,
        event_id: Optional[uuid.UUID] = None,
        zone_id: Optional[uuid.UUID] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        async with self:
            query = select(Alert)
            if event_id:
                query = query.where(Alert.event_id == event_id)
            if zone_id:
                query = query.where(Alert.zone_id == zone_id)
            if status:
                query = query.where(Alert.status == AlertStatus(status))
            query = query.order_by(Alert.timestamp.desc()).limit(limit)
            result = await self._session.execute(query)
            return list(result.scalars().all())

    async def acknowledge(self, alert_id: uuid.UUID, operator_id: Optional[Any] = None) -> Optional[Alert]:
        async with self:
            alert = await self.get_by_id(alert_id)
            if not alert:
                return None
            op_uuid = None
            if operator_id:
                if isinstance(operator_id, uuid.UUID):
                    op_uuid = operator_id
                elif isinstance(operator_id, str):
                    try:
                        op_uuid = uuid.UUID(operator_id)
                    except ValueError:
                        op_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, operator_id)
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = op_uuid
            alert.acknowledged_at = datetime.now()
            await self._session.flush()
            return alert

    async def count_active(self, zone_id: Optional[uuid.UUID] = None) -> int:
        async with self:
            query = select(func.count(Alert.id)).where(Alert.status == AlertStatus.ACTIVE)
            if zone_id:
                query = query.where(Alert.zone_id == zone_id)
            result = await self._session.execute(query)
            return result.scalar() or 0

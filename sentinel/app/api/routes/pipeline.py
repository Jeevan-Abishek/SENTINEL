from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import asyncio
import json

from app.pipeline import PipelineConfig, pipeline_manager
from app.video import FileVideoAdapter, RTSPVideoAdapter, MockVideoAdapter
from app.detection.models import DetectorConfig
from app.tracking.bytetrack import TrackerConfig
from app.analytics.models import AnalyticsConfig
from app.analytics.zone_config import ZoneOccupancyConfig
from app.risk.models import RiskConfig
from app.alerts.manager import AlertConfig
from app.mapping import CoordinateCalibrator, ZoneManager


router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


class VideoSourceConfig(BaseModel):
    type: str = Field(..., description="file, rtsp, or mock")
    path: Optional[str] = None  # For file
    rtsp_url: Optional[str] = None  # For RTSP
    mock_fps: float = 30.0  # For mock
    mock_width: int = 1920  # For mock
    mock_height: int = 1080  # For mock


class PipelineCreateRequest(BaseModel):
    event_id: str
    video_source: VideoSourceConfig
    detection_config: Optional[DetectorConfig] = None
    tracker_config: Optional[TrackerConfig] = None
    analytics_config: Optional[AnalyticsConfig] = None
    zone_configs: List[ZoneOccupancyConfig] = []
    risk_config: Optional[RiskConfig] = None
    alert_config: Optional[AlertConfig] = None
    enable_analytics: bool = True
    enable_risk: bool = True
    enable_alerts: bool = True
    target_fps: float = 30.0


class PipelineResponse(BaseModel):
    pipeline_id: str
    event_id: str
    status: str
    stats: Dict[str, Any]


class PipelineStatusResponse(BaseModel):
    pipeline_id: str
    event_id: str
    status: str
    stats: Dict[str, Any]


# Store pipeline configs
_pipeline_configs: Dict[str, PipelineCreateRequest] = {}


def create_video_adapter(config: VideoSourceConfig):
    if config.type == "file":
        if not config.path:
            raise ValueError("File path required for file source")
        return FileVideoAdapter(config.path)
    elif config.type == "rtsp":
        if not config.rtsp_url:
            raise ValueError("RTSP URL required for RTSP source")
        return RTSPVideoAdapter(config.rtsp_url)
    elif config.type == "mock":
        return MockVideoAdapter(
            fps=config.mock_fps,
            width=config.mock_width,
            height=config.mock_height,
        )
    else:
        raise ValueError(f"Unknown video source type: {config.type}")


@router.post("", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(request: PipelineCreateRequest):
    pipeline_id = str(uuid.uuid4())
    
    # Create video adapter
    video_adapter = create_video_adapter(request.video_source)
    
# Use default configs if not provided
    detection_config = request.detection_config or DetectorConfig()
    tracker_config = request.tracker_config or TrackerConfig()
    analytics_config = request.analytics_config or AnalyticsConfig()
    risk_config = request.risk_config or RiskConfig()
    alert_config = request.alert_config or AlertConfig()
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        event_id=request.event_id,
        video_adapter=video_adapter,
        detection_config=detection_config,
        tracker_config=tracker_config,
        analytics_config=analytics_config,
        zone_configs=request.zone_configs,
        risk_config=risk_config,
        alert_config=alert_config,
        enable_analytics=request.enable_analytics,
        enable_risk=request.enable_risk,
        enable_alerts=request.enable_alerts,
        target_fps=request.target_fps,
    )
    
    # Create pipeline
    processor = pipeline_manager.create_pipeline(pipeline_id, pipeline_config)
    _pipeline_configs[pipeline_id] = request
    
    return PipelineResponse(
        pipeline_id=pipeline_id,
        event_id=request.event_id,
        status="created",
        stats=processor.get_stats(),
    )


@router.get("", response_model=List[PipelineResponse])
async def list_pipelines():
    result = []
    for pid, processor in pipeline_manager.pipelines.items():
        config = _pipeline_configs.get(pid)
        result.append(PipelineResponse(
            pipeline_id=pid,
            event_id=config.event_id if config else "unknown",
            status="running" if processor._running else "stopped",
            stats=processor.get_stats(),
        ))
    return result


@router.get("/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline(pipeline_id: str):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    config = _pipeline_configs.get(pipeline_id)
    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        event_id=config.event_id if config else "unknown",
        status="running" if processor._running else "stopped",
        stats=processor.get_stats(),
    )


@router.post("/{pipeline_id}/start", response_model=PipelineStatusResponse)
async def start_pipeline(pipeline_id: str):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    if processor._running:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    await processor.start()
    
    config = _pipeline_configs.get(pipeline_id)
    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        event_id=config.event_id if config else "unknown",
        status="running",
        stats=processor.get_stats(),
    )


@router.post("/{pipeline_id}/stop", response_model=PipelineStatusResponse)
async def stop_pipeline(pipeline_id: str):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    if not processor._running:
        raise HTTPException(status_code=400, detail="Pipeline not running")
    
    await processor.stop()
    
    config = _pipeline_configs.get(pipeline_id)
    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        event_id=config.event_id if config else "unknown",
        status="stopped",
        stats=processor.get_stats(),
    )


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(pipeline_id: str):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if processor and processor._running:
        await processor.stop()
    
    pipeline_manager.remove_pipeline(pipeline_id)
    if pipeline_id in _pipeline_configs:
        del _pipeline_configs[pipeline_id]


@router.websocket("/{pipeline_id}/ws")
async def pipeline_websocket(websocket: WebSocket, pipeline_id: str):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        await websocket.close(code=4004, reason="Pipeline not found")
        return
    
    await websocket.accept()
    
    try:
        while True:
            result = await processor.get_result()
            if result:
                # Convert to JSON-serializable format
                data = {
                    "frame_id": result.frame_id,
                    "timestamp": result.timestamp,
                    "detections_count": len(result.detections.detections),
                    "tracks_count": len(result.tracks.tracks),
                }
                
                if result.count:
                    data["count"] = {
                        "total": result.count.total_count,
                        "current": result.count.current_count,
                        "in": result.count.in_count,
                        "out": result.count.out_count,
                    }
                
                if result.density:
                    data["density"] = {
                        "avg_density": result.density.avg_density,
                        "max_density": result.density.max_density,
                        "grid_shape": result.density.grid_density.shape if result.density.grid_density is not None else None,
                    }
                
                if result.occupancy:
                    data["occupancy"] = [
                        {
                            "zone_id": o.zone_id,
                            "count": o.current_count,
                            "capacity": o.capacity,
                            "occupancy_ratio": o.occupancy_ratio,
                            "status": o.status,
                        }
                        for o in result.occupancy
                    ]
                
                if result.risk:
                    data["risk"] = {
                        "overall_score": result.risk.overall_score,
                        "risk_level": result.risk.risk_level.value,
                        "recommended_actions": result.risk.recommended_actions,
                    }
                
                if result.alerts:
                    data["alerts"] = [
                        {
                            "id": a.id,
                            "zone_id": a.zone_id,
                            "risk_level": a.risk_level,
                            "message": a.message,
                            "risk_score": a.risk_score,
                        }
                        for a in result.alerts
                    ]
                
                await websocket.send_json(data)
            else:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        pass


@router.post("/{pipeline_id}/calibration", status_code=status.HTTP_200_OK)
async def set_pipeline_calibration(pipeline_id: str, calibration_data: Dict):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    calibrator = CoordinateCalibrator()
    calibrator.load_calibration_data(calibration_data)
    processor.config.calibration = calibrator
    
    return {"status": "calibration_set"}


@router.post("/{pipeline_id}/zones", status_code=status.HTTP_200_OK)
async def set_pipeline_zones(pipeline_id: str, zones_data: Dict):
    processor = pipeline_manager.get_pipeline(pipeline_id)
    if not processor:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    zone_manager = ZoneManager()
    if processor.config.calibration:
        zone_manager.set_calibrator(processor.config.calibration)
    
    for zone_id, zone_info in zones_data.items():
        zone_manager.add_zone(
            zone_id=zone_id,
            name=zone_info.get("name", zone_id),
            zone_type=zone_info.get("zone_type", "monitoring"),
            image_polygon=zone_info.get("image_polygon", []),
            capacity=zone_info.get("capacity", 0),
            entry_line=zone_info.get("entry_line"),
            exit_line=zone_info.get("exit_line"),
        )
    
    processor.config.zone_manager = zone_manager
    
    return {"status": "zones_set", "zone_count": len(zones_data)}
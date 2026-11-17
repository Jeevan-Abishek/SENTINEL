import asyncio
import cv2
import numpy as np
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import time
import uuid

from app.video import (
    FileVideoAdapter,
    RTSPVideoAdapter,
    MockVideoAdapter,
    FramePreprocessor,
    FrameQualityChecker,
    ProcessedFrame,
    VideoFrame,
)
from app.detection.yolo_detector import YOLODetector
from app.detection.models import Detection, DetectionResult, DetectorConfig
from app.detection.detector import DetectionContext
from app.tracking.tracker import Tracker
from app.tracking.models import Track, TrackResult, TrackerConfig, TrackingContext
from app.analytics import (
    PeopleCounter,
    CountingResult,
    DensityEstimator,
    DensityResult,
    AnalyticsConfig,
    OccupancyMonitor,
    OccupancyResult,
    MovementAnalyzerWrapper,
    MovementResult,
    FlowAnalyzerWrapper,
    FlowResult,
    BottleneckDetectorWrapper,
    BottleneckResult,
)
from app.analytics.zone_config import ZoneOccupancyConfig
from app.analytics.models import AnalyticsContext
from app.risk.engine import RiskEngineWrapper
from app.risk.models import RiskConfig, RiskAssessment, AlertConfig, Alert
from app.alerts.manager import AlertManagerWrapper
from app.alerts.notifications import WebSocketNotificationChannel, SoundNotificationChannel
from app.mapping import CoordinateCalibrator, ZoneManager


@dataclass
class PipelineConfig:
    event_id: str
    video_adapter: Any  # FileVideoAdapter, RTSPVideoAdapter, or MockVideoAdapter
    detection_config: DetectorConfig
    tracker_config: TrackerConfig
    analytics_config: AnalyticsConfig
    zone_configs: List[ZoneOccupancyConfig]
    risk_config: RiskConfig
    alert_config: AlertConfig
    calibration: Optional[CoordinateCalibrator] = None
    zone_manager: Optional[ZoneManager] = None
    enable_analytics: bool = True
    enable_risk: bool = True
    enable_alerts: bool = True
    target_fps: float = 30.0


@dataclass
class PipelineResult:
    frame_id: int
    timestamp: float
    detections: DetectionResult
    tracks: TrackResult
    count: Optional[Dict[str, CountingResult]] = None
    density: Optional[Dict[str, DensityResult]] = None
    occupancy: Optional[Dict[str, OccupancyResult]] = None
    movement: Optional[Dict[str, MovementResult]] = None
    flow: Optional[Dict[str, FlowResult]] = None
    bottleneck: Optional[Dict[str, BottleneckResult]] = None
    risk: Optional[RiskAssessment] = None
    alerts: Optional[List[Alert]] = None
    quality: Optional[Any] = None  # QualityMetrics
    processing_time_ms: float = 0.0


class VideoStreamProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.frame_preprocessor = FramePreprocessor()
        self.quality_checker = FrameQualityChecker()
        
        # Initialize detector
        self.detector = YOLODetector(config.detection_config)
        
        # Initialize tracker
        self.tracker = Tracker(config.tracker_config)
        
        # Initialize analytics with config
        self.analytics_config = config.analytics_config
        self.counter = None  # Will be initialized per zone
        self.density_estimator = DensityEstimator(config.analytics_config)
        self.occupancy_monitor = OccupancyMonitor(config.analytics_config)
        self.movement_analyzer = MovementAnalyzerWrapper(config.analytics_config)
        self.flow_analyzer = FlowAnalyzerWrapper(config.analytics_config)
        self.bottleneck_detector = BottleneckDetectorWrapper(config.analytics_config)
        
        # Initialize risk engine
        self.risk_engine = RiskEngineWrapper(config.risk_config)
        
        # Initialize alert manager
        self.alert_manager = AlertManagerWrapper(config.alert_config)
        
        # Setup notification channels
        self.websocket_channel = WebSocketNotificationChannel(ws_manager=None)
        self.sound_channel = SoundNotificationChannel(sound_files={})
        self.alert_manager.register_notification_handler(self.websocket_channel.send)
        self.alert_manager.register_notification_handler(self.sound_channel.send)
        
        # State
        self._running = False
        self._frame_id = 0
        self._frame_queue: Optional[asyncio.Queue] = None
        self._result_queue: Optional[asyncio.Queue] = None
        self._process_task: Optional[asyncio.Task] = None
        self._capture_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_result: Optional[Callable[[PipelineResult], None]] = None
        self.on_alert: Optional[Callable[[Alert], None]] = None
        self.on_frame: Optional[Callable[[np.ndarray], None]] = None
        
        # Stats
        self.stats = {
            "frames_processed": 0,
            "frames_dropped": 0,
            "avg_processing_time": 0.0,
            "avg_fps": 0.0,
            "last_frame_time": 0.0,
        }

    async def start(self):
        if self._running:
            return
        
        self._running = True
        self._frame_queue = asyncio.Queue(maxsize=10)
        self._result_queue = asyncio.Queue(maxsize=10)
        await self.config.video_adapter.connect()
        
        # Load detection model
        self.detector.load_model()
        
        # Start capture and process tasks
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._process_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
        
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        await self.config.video_adapter.disconnect()

    async def _capture_loop(self):
        frame_interval = 1.0 / self.config.target_fps
        
        while self._running:
            start_time = time.time()
            
            try:
                frame = await self.config.video_adapter.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Check quality
                quality = self.quality_checker.check_quality(frame.frame)
                if not quality.get("passed", True):
                    self.stats["frames_dropped"] += 1
                    continue
                
                # Preprocess
                processed_frame_obj = self.frame_preprocessor.preprocess(frame.frame)
                processed_frame = processed_frame_obj.processed
                
                # Try to put in queue (non-blocking)
                try:
                    self._frame_queue.put_nowait({
                        "frame": processed_frame,
                        "original": frame.frame,
                        "quality": quality,
                        "timestamp": time.time(),
                        "frame_id": self._frame_id,
                    })
                    self._frame_id += 1
                except asyncio.QueueFull:
                    self.stats["frames_dropped"] += 1
                
                # Frame rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Capture error: {e}")
                await asyncio.sleep(0.1)

    async def _process_loop(self):
        while self._running:
            try:
                # Get frame from queue with timeout
                frame_data = await asyncio.wait_for(self._frame_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            
            process_start = time.time()
            
            try:
                result = await self._process_frame(frame_data)
                
                # Update stats
                process_time = (time.time() - process_start) * 1000
                self.stats["frames_processed"] += 1
                self.stats["avg_processing_time"] = (
                    0.9 * self.stats["avg_processing_time"] + 0.1 * process_time
                )
                self.stats["last_frame_time"] = time.time()
                
                # Put result in output queue
                try:
                    self._result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    pass  # Drop oldest if queue full
                
                # Call callback
                if self.on_result:
                    self.on_result(result)
                    
            except Exception as e:
                print(f"Processing error: {e}")

    def _create_analytics_context(self, zone_config: ZoneOccupancyConfig) -> AnalyticsContext:
        """Create an AnalyticsContext for a zone."""
        world_polygon = None
        homography = None
        
        if self.config.zone_manager and zone_config.zone_id in self.config.zone_manager.zones:
            zone = self.config.zone_manager.zones[zone_config.zone_id]
            world_polygon = getattr(zone, "world_polygon", None) or getattr(zone, "image_polygon", None)
            if not world_polygon and isinstance(zone, dict):
                world_polygon = zone.get("world_polygon") or zone.get("image_polygon")

        if not world_polygon:
            world_polygon = zone_config.polygon
        
        if self.config.calibration and self.config.calibration.homography_matrix is not None:
            homography = self.config.calibration.homography_matrix.tolist()
        
        return AnalyticsContext(
            zone_id=zone_config.zone_id,
            frame_number=self._frame_id,
            timestamp=time.time(),
            zone_polygon=world_polygon,
            zone_capacity=zone_config.capacity,
            homography_matrix=homography,
            cell_size_meters=self.config.analytics_config.density_grid_size,
        )

    async def _process_frame(self, frame_data: Dict) -> PipelineResult:
        processed_frame = frame_data["frame"]
        original_frame = frame_data["original"]
        timestamp = frame_data["timestamp"]
        frame_id = frame_data["frame_id"]
        quality = frame_data["quality"]
        
        # Detection
        detection_context = DetectionContext(
            frame_number=frame_id,
            timestamp=timestamp,
            original_frame_shape=original_frame.shape[:2],
        )
        detections = self.detector.detect(original_frame, detection_context)
        
        # Tracking
        tracking_context = TrackingContext(
            frame_number=frame_id,
            timestamp=timestamp,
            original_frame_shape=original_frame.shape[:2],
        )
        tracks = self.tracker.update(detections, tracking_context)
        
        # Analytics per zone
        count = {}
        density = {}
        occupancy = {}
        movement = {}
        flow = {}
        bottleneck = {}
        
        if self.config.enable_analytics and self.config.zone_configs:
            for zone_config in self.config.zone_configs:
                zone_id = zone_config.zone_id
                context = self._create_analytics_context(zone_config)
                
                # People counting
                counter = PeopleCounter(zone_id, context.zone_polygon)
                count[zone_id] = counter.update(tracks.tracks, context, self.config.target_fps)
                
                # Density estimation
                density[zone_id] = self.density_estimator.estimate(tracks.tracks, context, self.config.target_fps)
                
                # Occupancy monitoring
                occupancy[zone_id] = self.occupancy_monitor.analyze(
                    count[zone_id].current_count,
                    zone_config.capacity,
                    timestamp,
                    zone_id,
                )
                
                # Movement analysis
                movement[zone_id] = self.movement_analyzer.analyze(tracks.tracks, context, self.config.target_fps)
                
                # Flow analysis
                entry_lines = []
                exit_lines = []
                if self.config.zone_manager and zone_id in self.config.zone_manager.zones:
                    zone = self.config.zone_manager.zones[zone_id]
                    el = getattr(zone, "world_entry_line", None) or getattr(zone, "entry_line", None) or (zone.get("world_entry_line") if isinstance(zone, dict) else None)
                    xl = getattr(zone, "world_exit_line", None) or getattr(zone, "exit_line", None) or (zone.get("world_exit_line") if isinstance(zone, dict) else None)
                    if el and len(el) >= 2:
                        entry_lines = [{"start": el[0], "end": el[1]}]
                    if xl and len(xl) >= 2:
                        exit_lines = [{"start": xl[0], "end": xl[1]}]
                
                flow[zone_id] = self.flow_analyzer.analyze(
                    tracks.tracks, context, self.config.target_fps, entry_lines, exit_lines
                )
                
                # Bottleneck detection
                bottleneck[zone_id] = self.bottleneck_detector.analyze(
                    density[zone_id], flow[zone_id], movement[zone_id], context
                )
        
        # Risk assessment
        risk = None
        alerts = []
        
        if self.config.enable_risk and self.config.zone_configs:
            # Assess risk per zone
            risk_dict = {}
            for zone_config in self.config.zone_configs:
                zone_id = zone_config.zone_id
                zone_risk = self.risk_engine.assess(
                    zone_id=zone_id,
                    timestamp=timestamp,
                    density_result=density.get(zone_id),
                    occupancy_result=occupancy.get(zone_id),
                    movement_result=movement.get(zone_id),
                    flow_result=flow.get(zone_id),
                    bottleneck_result=bottleneck.get(zone_id),
                )
                risk_dict[zone_id] = zone_risk
            
            # For now, use the highest risk zone as overall risk
            if risk_dict:
                risk = max(risk_dict.values(), key=lambda r: r.risk_score)
            
            # Alert management
            if self.config.enable_alerts:
                for zone_config in self.config.zone_configs:
                    zone_id = zone_config.zone_id
                    zone_risk = risk_dict.get(zone_id)
                    if zone_risk:
                        zone_alerts = self.alert_manager.check_alerts(zone_risk)
                        alerts.extend(zone_alerts)
                        for alert in zone_alerts:
                            if self.on_alert:
                                self.on_alert(alert)
        
        return PipelineResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            tracks=tracks,
            count=count,
            density=density,
            occupancy=occupancy,
            movement=movement,
            flow=flow,
            bottleneck=bottleneck,
            risk=risk,
            alerts=alerts,
            quality=quality,
            processing_time_ms=(time.time() - timestamp) * 1000,
        )

    async def get_result(self) -> Optional[PipelineResult]:
        if not self._result_queue:
            return None
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> Dict:
        elapsed = time.time() - self.stats["last_frame_time"] if self.stats["last_frame_time"] > 0 else 1
        if self.stats["frames_processed"] > 0:
            self.stats["avg_fps"] = self.stats["frames_processed"] / elapsed
        return self.stats.copy()


class PipelineManager:
    def __init__(self):
        self.pipelines: Dict[str, VideoStreamProcessor] = {}
    
    def create_pipeline(self, pipeline_id: str, config: PipelineConfig) -> VideoStreamProcessor:
        processor = VideoStreamProcessor(config)
        self.pipelines[pipeline_id] = processor
        return processor
    
    def get_pipeline(self, pipeline_id: str) -> Optional[VideoStreamProcessor]:
        return self.pipelines.get(pipeline_id)
    
    def remove_pipeline(self, pipeline_id: str):
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]
    
    async def start_all(self):
        for pipeline in self.pipelines.values():
            await pipeline.start()
    
    async def stop_all(self):
        for pipeline in self.pipelines.values():
            await pipeline.stop()
        self.pipelines.clear()


# Global pipeline manager instance
pipeline_manager = PipelineManager()
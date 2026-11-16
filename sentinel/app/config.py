from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "SENTINEL"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Database
    database_url: str = Field(..., description="PostgreSQL async connection URL")
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    access_token_expire_minutes: int = 30

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501

    # Video Processing
    default_confidence_threshold: float = 0.5
    default_iou_threshold: float = 0.45
    max_detections_per_frame: int = 1000
    frame_skip: int = 1
    processing_width: int = 640
    processing_height: int = 640

    # YOLO Model
    yolo_model_path: str = "models/yolov8n.pt"
    yolo_device: Literal["cpu", "cuda", "mps"] = "cpu"
    yolo_half_precision: bool = False

    # Tracking
    tracker_type: Literal["bytetrack", "botsort"] = "bytetrack"
    track_buffer: int = 30
    match_threshold: float = 0.8
    min_box_area: int = 10

    # Re-ID (Optional)
    reid_enabled: bool = False
    reid_model_path: str = "models/osnet_x1_0.pth"
    reid_threshold: float = 0.6

    # Analytics
    density_grid_size: float = 1.0
    occupancy_warning_threshold: float = 0.7
    occupancy_critical_threshold: float = 0.9
    flow_window_seconds: int = 60
    bottleneck_persistence_frames: int = 30

    # Risk Engine
    risk_density_weight: float = 0.3
    risk_occupancy_weight: float = 0.2
    risk_movement_weight: float = 0.2
    risk_flow_weight: float = 0.15
    risk_bottleneck_weight: float = 0.15
    risk_safe_threshold: float = 30.0
    risk_warning_threshold: float = 60.0
    risk_critical_threshold: float = 80.0
    risk_temporal_smoothing: float = 0.3

    # Alerts
    alert_cooldown_seconds: int = 30
    alert_max_active: int = 100
    alert_sound_enabled: bool = True

    # Drone (Mock for development)
    drone_mock_mode: bool = True
    drone_rtsp_url: Optional[str] = None
    drone_telemetry_port: int = 14550

    # Mapping
    default_map_center_lat: float = 0.0
    default_map_center_lon: float = 0.0
    default_map_zoom: int = 18

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "data" / "models"

    @property
    def sample_videos_dir(self) -> Path:
        return self.base_dir / "data" / "sample_videos"

    @property
    def resolved_yolo_model_path(self) -> Path:
        path = Path(self.yolo_model_path)
        if not path.is_absolute():
            return self.models_dir / path.name
        return path

    @property
    def resolved_reid_model_path(self) -> Path:
        path = Path(self.reid_model_path)
        if not path.is_absolute():
            return self.models_dir / path.name
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
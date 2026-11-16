"""
Unit tests for SENTINEL configuration module.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import Settings, get_settings


class TestSettings:
    """Tests for Settings configuration."""
    
    def test_default_values(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        assert settings.app_name == "SENTINEL"
        assert settings.app_env == "development"
        assert settings.debug is True
        assert settings.log_level == "INFO"
    
    def test_database_config(self):
        settings = Settings(
            database_url="postgresql+asyncpg://user:pass@localhost/db",
            secret_key="test-secret-key"
        )
        
        assert settings.database_url == "postgresql+asyncpg://user:pass@localhost/db"
        assert settings.database_pool_size == 10
        assert settings.database_max_overflow == 20
    
    def test_video_processing_config(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        assert settings.default_confidence_threshold == 0.5
        assert settings.default_iou_threshold == 0.45
        assert settings.frame_skip == 1
        assert settings.processing_width == 640
        assert settings.processing_height == 640
    
    def test_yolo_config(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        assert settings.yolo_model_path == "models/yolov8n.pt"
        assert settings.yolo_device == "cpu"
        assert settings.yolo_half_precision is False
    
    def test_tracker_config(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        assert settings.tracker_type == "bytetrack"
        assert settings.track_buffer == 30
        assert settings.match_threshold == 0.8
    
    def test_risk_engine_config(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        assert settings.risk_density_weight == 0.3
        assert settings.risk_occupancy_weight == 0.2
        assert settings.risk_movement_weight == 0.2
        assert settings.risk_flow_weight == 0.15
        assert settings.risk_bottleneck_weight == 0.15
        assert settings.risk_safe_threshold == 30.0
        assert settings.risk_warning_threshold == 60.0
        assert settings.risk_critical_threshold == 80.0
    
    def test_properties(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        # base_dir returns the project root (parent of app/)
        assert settings.base_dir.name == "sentinel"
        assert settings.models_dir.name == "models"
        assert settings.sample_videos_dir.name == "sample_videos"
    
    def test_resolved_model_paths(self):
        settings = Settings(
            database_url="sqlite+aiosqlite:///./test.db",
            secret_key="test-secret-key"
        )
        
        yolo_path = settings.resolved_yolo_model_path
        assert yolo_path.name == "yolov8n.pt"
        assert yolo_path.parent.name == "models"
        
        reid_path = settings.resolved_reid_model_path
        assert reid_path.name == "osnet_x1_0.pth"
        assert reid_path.parent.name == "models"


class TestGetSettings:
    """Tests for get_settings function."""
    
    def test_get_settings_singleton(self):
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2  # Should be cached
    
    def test_get_settings_returns_settings(self):
        settings = get_settings()
        assert isinstance(settings, Settings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
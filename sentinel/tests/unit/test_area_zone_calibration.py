"""Unit tests for Area & Zone Calibration, Scale Factors, and Dynamic Thresholds."""

import pytest
import math
import numpy as np
from pathlib import Path

from app.mapping.calibration import (
    ScaleCalibrator,
    ScaleCalibrationResult,
    CoordinateCalibrator,
    CameraFOVParams,
)
from app.mapping.zones import (
    ZoneDefinition,
    ZoneManager,
)


class TestScaleCalibrator:
    def test_from_two_points(self):
        # 100 pixels represents 10.0 meters -> scale = 0.1 m/px
        pt1 = (100.0, 100.0)
        pt2 = (200.0, 100.0)
        res = ScaleCalibrator.from_two_points(pt1, pt2, real_distance_m=10.0)

        assert res.is_valid is True
        assert pytest.approx(res.scale_m_per_px, abs=1e-4) == 0.10
        assert pytest.approx(res.area_m2_per_pixel, abs=1e-5) == 0.01
        assert res.pixel_distance_px == 100.0
        assert res.reference_distance_m == 10.0

    def test_from_two_points_invalid(self):
        # Zero distance or identical points
        pt1 = (50.0, 50.0)
        pt2 = (50.0, 50.0)
        res = ScaleCalibrator.from_two_points(pt1, pt2, real_distance_m=0.0)
        assert res.is_valid is False
        assert res.scale_m_per_px == 0.0

    def test_from_altitude(self):
        params = CameraFOVParams(hfov_deg=84.0, vfov_deg=56.0, image_width=1920, image_height=1080)
        res = ScaleCalibrator.from_altitude(altitude_m=30.0, params=params)

        assert res.is_valid is True
        assert res.scale_m_per_px > 0
        assert res.area_m2_per_pixel > 0
        assert res.method == "altitude_gsd"

    def test_calculate_polygon_area_from_scale(self):
        # A 100x100 pixel square with scale = 0.1 m/px -> 10m x 10m = 100 m²
        polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        area = ScaleCalibrator.calculate_polygon_area_from_scale(polygon, scale_m_per_px=0.1)

        assert area is not None
        assert pytest.approx(area, abs=0.1) == 100.0


class TestCoordinateCalibratorScaleIntegration:
    def test_two_point_scale_integration(self):
        calibrator = CoordinateCalibrator()
        # Calibrate 100 pixels = 20 meters -> 0.2 m/px
        calibrator.set_two_point_scale((0.0, 0.0), (100.0, 0.0), real_distance_m=20.0)
        assert calibrator.two_point_scale is not None
        assert calibrator.two_point_scale.scale_m_per_px == 0.20

        # 50x50 pixel rectangle -> 10m x 10m = 100 m²
        poly = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
        area = calibrator.calculate_polygon_ground_area(poly)
        assert area is not None
        assert pytest.approx(area, abs=0.1) == 100.0

    def test_explicit_scale_override(self):
        calibrator = CoordinateCalibrator()
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        # Override with scale 0.05 m/px -> 5m x 5m = 25 m²
        area = calibrator.calculate_polygon_ground_area(poly, scale_m_per_px=0.05)
        assert area is not None
        assert pytest.approx(area, abs=0.1) == 25.0


class TestZoneCalibrationAndThresholds:
    def test_zone_custom_thresholds_and_dict(self):
        zone = ZoneDefinition(
            zone_id="choke_south",
            name="South Choke Tunnel",
            zone_type="choke_point",
            image_polygon=[(0, 0), (20, 0), (20, 10), (0, 10)],
            capacity=50,
            area_m2=25.0,
            warning_density_p_sqm=1.2,
            critical_density_p_sqm=2.4,
            warning_occupancy_ratio=0.50,
            critical_occupancy_ratio=0.80,
            risk_weight=1.5,
            notes="High priority bottleneck",
        )

        d = zone.to_dict()
        assert d["zone_id"] == "choke_south"
        assert d["warning_density_p_sqm"] == 1.2
        assert d["critical_density_p_sqm"] == 2.4
        assert d["risk_weight"] == 1.5

        # Reconstruct
        restored = ZoneDefinition.from_dict(d)
        assert restored.zone_id == "choke_south"
        assert restored.area_sqm == 25.0
        assert restored.warning_density_p_sqm == 1.2
        assert restored.critical_density_p_sqm == 2.4
        assert restored.risk_weight == 1.5

    def test_zone_manager_update_calibration(self):
        manager = ZoneManager()
        manager.add_rectangular_zone(
            zone_id="ingress_main",
            name="Main Ingress",
            zone_type="ingress",
            center_x=100, center_y=100,
            width=50, height=50,
            capacity=100, area_m2=50.0,
        )

        # Update calibration
        updated = manager.update_zone_calibration(
            zone_id="ingress_main",
            name="Main Ingress North",
            area_sqm=75.0,
            max_capacity=150,
            warning_density_p_sqm=1.8,
            critical_density_p_sqm=3.5,
            risk_weight=1.2,
        )
        assert updated is True

        zone = manager.get_zone("ingress_main")
        assert zone.name == "Main Ingress North"
        assert zone.area_sqm == 75.0
        assert zone.max_capacity == 150
        assert zone.warning_density_p_sqm == 1.8
        assert zone.critical_density_p_sqm == 3.5
        assert zone.risk_weight == 1.2

    def test_custom_threshold_evaluation(self):
        manager = ZoneManager()
        manager.add_rectangular_zone(
            zone_id="sensitive_vip",
            name="VIP Restricted Concourse",
            zone_type="restricted",
            center_x=200, center_y=200,
            width=50, height=50,
            capacity=100, area_m2=50.0,
            warning_density_p_sqm=0.8,  # Triggers earlier than default 1.5
            critical_density_p_sqm=1.6, # Triggers earlier than default 3.0
        )

        # 30 people / 50 m² = 0.60 p/m² -> SAFE (< 0.8 and occ 30% < 65%)
        m_safe = manager.update_zone_metrics("sensitive_vip", count=30)
        assert m_safe["risk_status"] == "SAFE"

        # 45 people / 50 m² = 0.90 p/m² -> WARNING (>= 0.8)
        m_warn = manager.update_zone_metrics("sensitive_vip", count=45)
        assert m_warn["risk_status"] == "WARNING"

        # 85 people / 50 m² = 1.70 p/m² -> CRITICAL (>= 1.6)
        m_crit = manager.update_zone_metrics("sensitive_vip", count=85)
        assert m_crit["risk_status"] == "CRITICAL"


class TestZoneManagerPersistence:
    def test_json_export_and_import(self):
        manager = ZoneManager()
        manager.add_rectangular_zone(
            zone_id="zone_a",
            name="Zone Alpha",
            zone_type="ingress",
            center_x=100, center_y=100, width=40, height=40,
            capacity=80, area_m2=60.0,
        )
        manager.add_rectangular_zone(
            zone_id="zone_b",
            name="Zone Beta",
            zone_type="main_assembly",
            center_x=300, center_y=300, width=100, height=100,
            capacity=200, area_m2=250.0,
        )

        json_str = manager.export_calibrations_json()
        assert "Zone Alpha" in json_str
        assert "Zone Beta" in json_str

        # Import into fresh manager
        new_manager = ZoneManager()
        success = new_manager.import_calibrations_json(json_str)
        assert success is True
        assert len(new_manager.get_all_zones()) == 2
        assert new_manager.get_zone("zone_a").area_sqm == 60.0
        assert new_manager.get_zone("zone_b").max_capacity == 200

    def test_file_save_and_load(self, tmp_path):
        manager = ZoneManager()
        manager.add_rectangular_zone(
            zone_id="file_zone",
            name="File Test Zone",
            zone_type="egress",
            center_x=50, center_y=50, width=20, height=20,
            capacity=30, area_m2=25.0,
        )

        test_file = tmp_path / "test_zones.json"
        saved = manager.save_to_file(filepath=test_file)
        assert saved is True
        assert test_file.exists()

        new_manager = ZoneManager()
        loaded = new_manager.load_from_file(filepath=test_file)
        assert loaded is True
        assert new_manager.get_zone("file_zone") is not None
        assert new_manager.get_zone("file_zone").name == "File Test Zone"

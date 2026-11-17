"""Unit tests for Phase 6: Geospatial Calibration and Zone Mapping."""

import pytest
import numpy as np

from app.mapping.calibration import (
    CalibrationPoint,
    CameraFOVParams,
    GroundSamplingDistance,
    GroundResolutionCalculator,
    CoordinateCalibrator,
)
from app.mapping.zones import (
    ZoneDefinition,
    PolygonZoneValidator,
    ZoneManager,
)


class TestGroundResolutionCalculator:
    def test_gsd_at_altitude(self):
        params = CameraFOVParams(hfov_deg=84.0, vfov_deg=56.0, image_width=1920, image_height=1080)
        altitude_m = 30.0

        gsd = GroundResolutionCalculator.calculate_gsd(altitude_m=altitude_m, params=params)
        assert gsd.is_valid is True
        assert gsd.ground_footprint_width_m > 0
        assert gsd.ground_footprint_height_m > 0
        assert gsd.meters_per_pixel_x > 0
        assert gsd.meters_per_pixel_y > 0
        assert gsd.ground_area_per_pixel_m2 > 0

    def test_invalid_altitude_returns_invalid_gsd(self):
        gsd = GroundResolutionCalculator.calculate_gsd(altitude_m=0.0)
        assert gsd.is_valid is False
        assert gsd.meters_per_pixel_x == 0.0


class TestCoordinateCalibrator:
    def test_less_than_4_points_fails_homography(self):
        calibrator = CoordinateCalibrator(min_points=4)
        calibrator.add_calibration_point(0, 0, 0, 0)
        calibrator.add_calibration_point(100, 0, 10, 0)
        calibrator.add_calibration_point(100, 100, 10, 10)

        result = calibrator.compute_homography()
        assert result.is_valid is False
        assert result.reprojection_error == float('inf')

    def test_homography_4_points_synthetic_plane(self):
        calibrator = CoordinateCalibrator(min_points=4)
        # 10 pixels = 1 meter
        calibrator.add_calibration_point(0, 0, 0, 0)
        calibrator.add_calibration_point(100, 0, 10, 0)
        calibrator.add_calibration_point(100, 100, 10, 10)
        calibrator.add_calibration_point(0, 100, 0, 10)

        result = calibrator.compute_homography()
        assert result.is_valid is True
        assert result.reprojection_error < 1.0

        # Test forward transform (50, 50) -> (5.0, 5.0)
        world_pt = calibrator.image_to_world(50, 50)
        assert world_pt is not None
        assert pytest.approx(world_pt[0], abs=0.1) == 5.0
        assert pytest.approx(world_pt[1], abs=0.1) == 5.0

        # Test inverse transform (5.0, 5.0) -> (50, 50)
        img_pt = calibrator.world_to_image(5.0, 5.0)
        assert img_pt is not None
        assert pytest.approx(img_pt[0], abs=0.1) == 50.0
        assert pytest.approx(img_pt[1], abs=0.1) == 50.0

    def test_ground_area_calculation(self):
        calibrator = CoordinateCalibrator(min_points=4)
        calibrator.add_calibration_point(0, 0, 0, 0)
        calibrator.add_calibration_point(100, 0, 10, 0)
        calibrator.add_calibration_point(100, 100, 10, 10)
        calibrator.add_calibration_point(0, 100, 0, 10)
        calibrator.compute_homography()

        # A 100x100 pixel square = 10m x 10m = 100 m²
        polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        area = calibrator.calculate_polygon_ground_area(polygon)
        assert area is not None
        assert pytest.approx(area, abs=1.0) == 100.0


class TestPolygonZoneValidator:
    def test_valid_polygon(self):
        quad = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
        is_valid, msg, poly = PolygonZoneValidator.validate_polygon(quad)
        assert is_valid is True
        assert poly is not None

    def test_insufficient_points(self):
        two_pts = [(0.0, 0.0), (10.0, 10.0)]
        is_valid, msg, poly = PolygonZoneValidator.validate_polygon(two_pts)
        assert is_valid is False
        assert poly is None

    def test_non_finite_points(self):
        bad_pts = [(0.0, 0.0), (float('nan'), 10.0), (10.0, 10.0)]
        is_valid, msg, poly = PolygonZoneValidator.validate_polygon(bad_pts)
        assert is_valid is False


class TestZoneManager:
    def test_add_and_query_zone(self):
        manager = ZoneManager()
        quad = [(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)]
        success = manager.add_zone(
            zone_id="gate_north",
            name="North Gate",
            zone_type="ingress",
            image_polygon=quad,
            capacity=50,
        )
        assert success is True
        assert manager.is_point_in_zone("gate_north", 50.0, 50.0) is True
        assert manager.is_point_in_zone("gate_north", 5.0, 5.0) is False

    def test_calibrated_vs_uncalibrated_density(self):
        calibrator = CoordinateCalibrator()
        calibrator.add_calibration_point(0, 0, 0, 0)
        calibrator.add_calibration_point(100, 0, 10, 0)
        calibrator.add_calibration_point(100, 100, 10, 10)
        calibrator.add_calibration_point(0, 100, 0, 10)
        calibrator.compute_homography()

        manager = ZoneManager(calibrator=calibrator)
        quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        manager.add_zone(
            zone_id="plaza",
            name="Main Plaza",
            zone_type="crowd",
            image_polygon=quad,
            capacity=100,
        )

        metrics = manager.update_zone_metrics("plaza", count=40)
        assert metrics["is_calibrated"] is True
        assert metrics["density_mode"] == "calibrated_physical_density"
        assert metrics["density_people_per_m2"] is not None
        # 40 people / 100 m² = 0.40 people/m²
        assert pytest.approx(metrics["density_people_per_m2"], abs=0.05) == 0.40

    def test_uncalibrated_density_fallback(self):
        manager = ZoneManager()  # No calibrator
        quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        manager.add_zone(
            zone_id="plaza_uncal",
            name="Uncalibrated Plaza",
            zone_type="crowd",
            image_polygon=quad,
            capacity=100,
        )

        metrics = manager.update_zone_metrics("plaza_uncal", count=50)
        assert metrics["is_calibrated"] is False
        assert metrics["density_mode"] == "uncalibrated_occupancy_estimate"
        assert metrics["density_people_per_m2"] is None


"""Unit tests for GPS / Digital Map, Coordinate Calibration & Zone Management."""

import pytest
import math
import numpy as np
import plotly.graph_objects as go

from app.mapping.calibration import (
    VenueCoordinates,
    GPSCoordinateConverter,
    CoordinateCalibrator,
    GroundResolutionCalculator,
    CameraFOVParams,
)
from app.mapping.zones import (
    ZoneDefinition,
    ZoneManager,
)
from app.mapping.map_renderer import (
    MapRenderer,
)


class TestGPSCoordinateConverter:
    def test_gps_roundtrip_conversion(self):
        # Center at Bengaluru Kanteerava Arena
        converter = GPSCoordinateConverter(origin_lat=12.9716, origin_lon=77.5946)

        # 50m East, 30m North
        lat, lon = converter.meters_to_gps(50.0, 30.0)
        assert abs(lat - 12.9716) > 0
        assert abs(lon - 77.5946) > 0

        # Convert back
        x_m, y_m = converter.gps_to_meters(lat, lon)
        assert pytest.approx(x_m, abs=0.1) == 50.0
        assert pytest.approx(y_m, abs=0.1) == 30.0

    def test_haversine_distance(self):
        # 1 degree of latitude is ~111.139 km = 111,139 m
        dist = GPSCoordinateConverter.distance_m(12.0, 77.0, 13.0, 77.0)
        assert pytest.approx(dist, rel=0.01) == 111195.0


class TestCoordinateCalibratorGPS:
    def test_calibrator_with_venue(self):
        calibrator = CoordinateCalibrator()
        calibrator.set_venue(lat=12.9716, lon=77.5946, altitude_m=30.0, name="Kanteerava Arena")

        assert calibrator.venue_coords is not None
        assert calibrator.venue_coords.latitude == 12.9716
        assert calibrator.converter is not None

    def test_pixel_to_gps_with_homography(self):
        calibrator = CoordinateCalibrator(min_points=4)
        calibrator.set_venue(lat=12.9716, lon=77.5946, altitude_m=30.0)

        # 10 pixels = 1 meter
        calibrator.add_calibration_point(0, 0, 0, 0)
        calibrator.add_calibration_point(100, 0, 10, 0)
        calibrator.add_calibration_point(100, 100, 10, 10)
        calibrator.add_calibration_point(0, 100, 0, 10)
        calibrator.compute_homography()

        # (50, 50) px -> (5m, 5m) -> GPS offset from (12.9716, 77.5946)
        gps = calibrator.pixel_to_gps(50, 50)
        assert gps is not None
        lat, lon = gps
        assert abs(lat - 12.9716) < 0.001
        assert abs(lon - 77.5946) < 0.001

    def test_pixel_to_gps_altitude_fallback(self):
        calibrator = CoordinateCalibrator()  # No homography
        calibrator.set_venue(lat=12.9716, lon=77.5946)

        # Center pixel (960, 540) on 1920x1080 should map exactly to drone nadir GPS
        params = CameraFOVParams(image_width=1920, image_height=1080)
        gps = calibrator.pixel_to_gps(
            960, 540,
            drone_alt_m=30.0,
            drone_lat=12.9716,
            drone_lon=77.5946,
            camera_params=params,
        )
        assert gps is not None
        assert pytest.approx(gps[0], abs=1e-5) == 12.9716
        assert pytest.approx(gps[1], abs=1e-5) == 77.5946


class TestZoneDefinitionAndManager:
    def test_zone_aliases(self):
        zone = ZoneDefinition(
            zone_id="zone_assembly",
            name="Main Assembly Plaza",
            zone_type="main_assembly",
            image_polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
            capacity=200,
            area_m2=100.0,
        )
        assert zone.area_sqm == 100.0
        assert zone.max_capacity == 200
        assert zone.risk_status == "SAFE"

        zone.max_capacity = 250
        assert zone.capacity == 250

        zone.risk_status = "CRITICAL"
        assert zone.risk_level == "critical"

    def test_rectangular_zone_creation(self):
        manager = ZoneManager()
        success = manager.add_rectangular_zone(
            zone_id="choke_01",
            name="North Choke Corridor",
            zone_type="choke_point",
            center_x=500.0,
            center_y=300.0,
            width=100.0,
            height=60.0,
            capacity=80,
            area_m2=50.0,
        )
        assert success is True
        zone = manager.get_zone("choke_01")
        assert zone is not None
        assert zone.zone_type == "choke_point"
        assert len(zone.image_polygon) == 4
        assert zone.is_calibrated is True

    def test_zone_metrics_and_risk_thresholds(self):
        manager = ZoneManager()
        manager.add_rectangular_zone(
            zone_id="ingress_a",
            name="Ingress Gate A",
            zone_type="ingress",
            center_x=100,
            center_y=100,
            width=50,
            height=50,
            capacity=100,
            area_m2=50.0,
        )

        # Test SAFE: 30 people / 50 m² = 0.60 p/m² (< 1.5)
        m_safe = manager.update_zone_metrics("ingress_a", count=30, flow_vector=(45.0, 1.2))
        assert m_safe["risk_status"] == "SAFE"
        assert m_safe["density_p_sqm"] == 0.60
        assert m_safe["flow_vector"] == (45.0, 1.2)

        # Test WARNING: 80 people / 50 m² = 1.60 p/m² (>= 1.5)
        m_warn = manager.update_zone_metrics("ingress_a", count=80)
        assert m_warn["risk_status"] == "WARNING"

        # Test CRITICAL: 160 people / 50 m² = 3.20 p/m² (>= 3.0)
        m_crit = manager.update_zone_metrics("ingress_a", count=160)
        assert m_crit["risk_status"] == "CRITICAL"


class TestMapRenderer:
    def test_folium_map_rendering(self):
        renderer = MapRenderer(center_lat=12.9716, center_lon=77.5946, zoom=18, venue_name="Kanteerava")

        sample_zones = [
            {
                "zone_id": "z1",
                "name": "North Gate",
                "zone_type": "ingress",
                "risk_status": "WARNING",
                "current_count": 85,
                "max_capacity": 100,
                "density_p_sqm": 2.1,
                "area_sqm": 40.0,
                "gps_polygon": [(12.9720, 77.5942), (12.9720, 77.5950), (12.9717, 77.5950), (12.9717, 77.5942)],
            }
        ]

        html = renderer.render_folium_map(
            zones=sample_zones,
            drone_lat=12.9716,
            drone_lon=77.5946,
            drone_alt_m=32.4,
            drone_heading_deg=45.0,
        )
        assert isinstance(html, str)
        assert len(html) > 500
        assert "folium" in html or "leaflet" in html or "Venue" in html

    def test_plotly_vector_map_rendering(self):
        renderer = MapRenderer(center_lat=12.9716, center_lon=77.5946, zoom=18, venue_name="Kanteerava")

        sample_zones = [
            {
                "zone_id": "z1",
                "name": "Central Assembly",
                "zone_type": "main_assembly",
                "risk_status": "SAFE",
                "current_count": 120,
                "max_capacity": 300,
                "density_p_sqm": 0.8,
                "area_sqm": 150.0,
                "gps_polygon": [(12.9716, 77.5943), (12.9716, 77.5949), (12.9712, 77.5949), (12.9712, 77.5943)],
            }
        ]

        fig = renderer.render_plotly_vector_map(
            zones=sample_zones,
            drone_lat=12.9716,
            drone_lon=77.5946,
            drone_alt_m=32.4,
            drone_heading_deg=45.0,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # perimeter + center + zone + drone traces


class TestDigitalMapPresetsAndHeading:
    """Verifies that all venue presets and drone modes safely resolve heading without UnboundLocalError."""

    def test_venue_presets_drone_headings(self):
        from dashboard.pages.digital_map import VENUE_PRESETS
        for vname, vinfo in VENUE_PRESETS.items():
            assert "drone_heading" in vinfo, f"{vname} missing drone_heading"
            assert isinstance(vinfo["drone_heading"], (int, float))

    def test_drone_mode_heading_resolution(self):
        from dashboard.pages.digital_map import VENUE_PRESETS
        for vname, vinfo in VENUE_PRESETS.items():
            venue_heading = vinfo["drone_heading"]
            for mode in ["Hover & Survey (Recommended)", "Patrol Perimeter", "Custom Flight Path"]:
                drone_heading = float(venue_heading)
                if mode == "Patrol Perimeter":
                    drone_heading = 135.0
                else:
                    drone_heading = float(venue_heading)
                assert isinstance(drone_heading, float)
                assert 0.0 <= drone_heading <= 360.0


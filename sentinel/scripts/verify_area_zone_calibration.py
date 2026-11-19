"""Comprehensive Verification Script for Area & Zone Calibration, Scale Factors & Thresholds."""

import sys
import time
import math
import tracemalloc
import json
from pathlib import Path

# Add project root and sentinel root to path
_workspace = Path(__file__).resolve().parent.parent.parent
_sentinel = _workspace / "sentinel"
for p in [str(_workspace), str(_sentinel)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.mapping.calibration import (
    ScaleCalibrator,
    ScaleCalibrationResult,
    GroundResolutionCalculator,
    CameraFOVParams,
    CoordinateCalibrator,
)
from app.mapping.zones import (
    ZoneDefinition,
    ZoneManager,
)
from dashboard.pages.zone_management import (
    init_session_zones,
    get_active_zones_dict,
    save_zone_to_session,
    reset_zones_to_defaults,
)


def run_verification():
    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 80)
    print("SENTINEL AREA & ZONE CALIBRATION COMPREHENSIVE VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Scale Calibration & Polygon Ground Area
    # -------------------------------------------------------------------------
    print("\n[STEP 1/5] Verifying 2-Point Scale Reference & Ground Area Computation...")
    # Reference distance: 120 pixels represents 15.0 meters on ground
    pt_a = (100.0, 100.0)
    pt_b = (220.0, 100.0)
    real_dist_m = 15.0
    scale_res = ScaleCalibrator.from_two_points(pt_a, pt_b, real_dist_m)

    expected_scale = 15.0 / 120.0  # 0.125 m/px
    print(f"  • Reference Pixel Distance: {scale_res.pixel_distance_px:.1f} px")
    print(f"  • Known Ground Distance: {real_dist_m:.2f} m")
    print(f"  • Computed Scale Factor: {scale_res.scale_m_per_px:.4f} m/pixel (Expected: {expected_scale:.4f})")
    print(f"  • Ground Area per Pixel: {scale_res.area_m2_per_pixel:.6f} m²/px")
    assert abs(scale_res.scale_m_per_px - expected_scale) < 1e-5

    # Compute area of 200x160 px rectangular zone
    # 200 * 0.125 = 25m, 160 * 0.125 = 20m -> 25 * 20 = 500 m²
    zone_poly = [(0.0, 0.0), (200.0, 0.0), (200.0, 160.0), (0.0, 160.0)]
    calc_area = ScaleCalibrator.calculate_polygon_area_from_scale(zone_poly, scale_res.scale_m_per_px)
    print(f"  • Zone Pixel Polygon: 200×160 px")
    print(f"  • Computed Ground Area: {calc_area:.2f} m² (Expected: 500.00 m²)")
    assert abs(calc_area - 500.0) < 0.1
    print("  ✅ Step 1: 2-Point Scale Calibration and Shoelace Ground Area verified.")

    # -------------------------------------------------------------------------
    # STEP 2: Drone Altitude AGL & Optical GSD Scale
    # -------------------------------------------------------------------------
    print("\n[STEP 2/5] Verifying Drone Altitude AGL Scale Factor & GSD Footprint...")
    params = CameraFOVParams(hfov_deg=84.0, vfov_deg=56.0, image_width=1920, image_height=1080)
    alt_scale = ScaleCalibrator.from_altitude(altitude_m=32.4, params=params, gimbal_pitch_deg=-90.0)

    print(f"  • Drone Altitude: 32.4 m AGL (Nadir -90°)")
    print(f"  • Average Optical Scale: {alt_scale.scale_m_per_px:.4f} m/pixel")
    print(f"  • Ground Area / Pixel: {alt_scale.area_m2_per_pixel:.6f} m²/px")
    assert alt_scale.is_valid and alt_scale.scale_m_per_px > 0

    calibrator = CoordinateCalibrator()
    calibrator.set_two_point_scale(pt_a, pt_b, real_dist_m)
    area_via_calibrator = calibrator.calculate_polygon_ground_area(zone_poly)
    print(f"  • Area via Integrated CoordinateCalibrator: {area_via_calibrator:.2f} m²")
    assert abs(area_via_calibrator - 500.0) < 0.1
    print("  ✅ Step 2: Drone Altitude AGL scale and Calibrator integration verified.")

    # -------------------------------------------------------------------------
    # STEP 3: Zone Creation, Calibration & Dynamic Thresholds
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] Verifying Zone Calibration & Dynamic Threshold Sensitivity...")
    manager = ZoneManager()

    # Zone 1: North Ingress Gate (Calibrated 60 m², capacity 120, default thresholds)
    manager.add_rectangular_zone(
        zone_id="gate_north",
        name="North Ingress Gate",
        zone_type="ingress",
        center_x=100, center_y=100, width=50, height=50,
        capacity=120, area_m2=60.0,
        warning_density_p_sqm=1.5, critical_density_p_sqm=3.0,
    )

    # Zone 2: Sensitive VIP Choke (Calibrated 25 m², capacity 50, strict thresholds)
    manager.add_rectangular_zone(
        zone_id="choke_vip",
        name="VIP Choke Tunnel",
        zone_type="choke_point",
        center_x=300, center_y=300, width=30, height=30,
        capacity=50, area_m2=25.0,
        warning_density_p_sqm=1.0, critical_density_p_sqm=2.0,
        risk_weight=1.5,
    )

    # Test North Ingress under count = 75
    # density = 75 / 60 = 1.25 p/m² (< 1.5) & occ = 75/120 = 62.5% (< 65%) -> SAFE
    m1 = manager.update_zone_metrics("gate_north", count=75)
    print(f"  • Zone 'North Ingress': Count=75/120 | Density={m1['density_p_sqm']} p/m² -> Status: [{m1['risk_status']}]")
    assert m1["risk_status"] == "SAFE"

    # Test VIP Choke under count = 30
    # density = 30 / 25 = 1.20 p/m² (>= 1.0 warn threshold!) -> WARNING
    m2 = manager.update_zone_metrics("choke_vip", count=30)
    print(f"  • Zone 'VIP Choke' (Strict Thresholds): Count=30/50 | Density={m2['density_p_sqm']} p/m² -> Status: [{m2['risk_status']}]")
    assert m2["risk_status"] == "WARNING"

    # Test VIP Choke under surge count = 55
    # density = 55 / 25 = 2.20 p/m² (>= 2.0 crit threshold!) -> CRITICAL
    m3 = manager.update_zone_metrics("choke_vip", count=55)
    print(f"  • Zone 'VIP Choke' (Surge): Count=55/50 | Density={m3['density_p_sqm']} p/m² -> Status: [{m3['risk_status']}]")
    assert m3["risk_status"] == "CRITICAL"
    print("  ✅ Step 3: Zone creation, customized thresholds & dynamic risk evaluations verified.")

    # -------------------------------------------------------------------------
    # STEP 4: Serialization & File Persistence
    # -------------------------------------------------------------------------
    print("\n[STEP 4/5] Verifying Serialization & Disk Persistence...")
    temp_calib_path = Path("data/calibrations/temp_zones_calibration.json")
    saved = manager.save_to_file(filepath=temp_calib_path)
    assert saved is True and temp_calib_path.exists()

    json_size = temp_calib_path.stat().st_size
    print(f"  • Saved Calibration File: '{temp_calib_path}' ({json_size} bytes)")

    # Restore in fresh manager instance
    fresh_manager = ZoneManager()
    loaded = fresh_manager.load_from_file(filepath=temp_calib_path)
    assert loaded is True
    assert len(fresh_manager.get_all_zones()) == 2
    restored_vip = fresh_manager.get_zone("choke_vip")
    assert restored_vip.warning_density_p_sqm == 1.0
    assert restored_vip.critical_density_p_sqm == 2.0
    assert restored_vip.area_sqm == 25.0
    print(f"  • Restored Zone: '{restored_vip.name}' (Area: {restored_vip.area_sqm} m², Weight: {restored_vip.risk_weight})")

    # Cleanup temp test file
    try:
        temp_calib_path.unlink()
    except Exception:
        pass
    print("  ✅ Step 4: JSON serialization and file persistence round-trip verified.")

    # -------------------------------------------------------------------------
    # STEP 5: Memory Footprint & Low-RAM Execution Benchmark
    # -------------------------------------------------------------------------
    print("\n[STEP 5/5] Measuring Resource Utilization & Footprint...")
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / (1024 * 1024)
    total_time_s = time.perf_counter() - t_start

    print(f"  • Peak Memory Allocated: {peak_mb:.2f} MB")
    print(f"  • Low-RAM Windows Budget: ~145.0 MB")
    print(f"  • Execution Margin: {(145.0 - peak_mb):.2f} MB ({(peak_mb/145.0)*100:.1f}% budget used)")
    print(f"  • Total Calibration Execution Time: {total_time_s:.2f} seconds")

    assert peak_mb < 40.0, f"Memory exceeded limit: {peak_mb:.2f} MB"
    print("  ✅ Step 5: Resource utilization verified within low-RAM limits.")

    print("\n" + "=" * 80)
    print("ALL 5 AREA & ZONE CALIBRATION MODULES 100% VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()


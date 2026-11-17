# SENTINEL — Drone Camera & Telemetry Integration Guide

This guide describes how to connect commercial drones, live video feeds, and telemetry streams to SENTINEL.

---

## 1. Supported Video Ingestion Interfaces

### A. RTSP Live Video Stream
Sentinel supports H.264 / H.265 RTSP streams from commercial drone controllers (e.g. DJI Smart Controller Enterprise, Autel Live Deck, Skydio, or generic RTSP transmitters):

```python
from app.video.drone_adapter import DroneStreamAdapter

adapter = DroneStreamAdapter(
    video_source="rtsp://192.168.1.100:8554/live",
    frame_skip=1,
    enable_motion_compensation=True,
    mock_telemetry=False,
)
await adapter.connect()
```

### B. Recorded Drone Video (MP4 / MOV / AVI)
Used for post-mission incident analysis, operational training, and ground-truth testing:
```python
adapter = DroneStreamAdapter(
    video_source="data/sample_videos/demo_drone_crowd.mp4",
    frame_skip=1,
    enable_motion_compensation=True,
    mock_telemetry=True,  # Automatically simulates telemetry synchronized to video timestamps
)
```

---

## 2. Ingesting Real Drone Telemetry

When telemetry is available via drone SDKs (e.g., DJI Mobile SDK, MAVLink, Dronecode SDK):

```python
from datetime import datetime
from app.video.telemetry import DroneTelemetry, GPSCoordinate, GimbalOrientation, DroneStatus

telemetry_data = DroneTelemetry(
    timestamp=datetime.now(),
    position=GPSCoordinate(latitude=12.9716, longitude=77.5946, altitude=35.0),
    gimbal=GimbalOrientation(pitch=-88.0, roll=0.0, yaw=120.0),
    battery_level=86.0,
    flight_mode="POS_HOLD",
    status=DroneStatus.FLYING,
    is_simulated=False,
)

# Feed directly to the adapter
adapter.feed_external_telemetry(telemetry_data)
```

---

## 3. Ground Plane Calibration (Homography vs GSD)

Sentinel offers two methods to compute physical ground metrics:

### Method 1: Reference Point Homography (Recommended for High Precision)
Mark 4 known landmarks (e.g. gate corners, road markings) in camera pixels and real-world ground meters:
```python
from app.mapping import CoordinateCalibrator

calibrator = CoordinateCalibrator()
calibrator.add_calibration_point(image_x=100, image_y=100, world_x=0.0, world_y=0.0)
calibrator.add_calibration_point(image_x=1100, image_y=100, world_x=50.0, world_y=0.0)
calibrator.add_calibration_point(image_x=1100, image_y=600, world_x=50.0, world_y=25.0)
calibrator.add_calibration_point(image_x=100, image_y=600, world_x=0.0, world_y=25.0)
result = calibrator.compute_homography()
# Resulting density is computed in people / m²
```

### Method 2: Altitude & Camera Field-of-View (GSD Automatic Mode)
If reference points are unavailable, Sentinel calculates ground sampling distance from drone altitude and camera sensor FOV:
$$\text{GSD} = \frac{2 \cdot \text{Altitude} \cdot \tan(\text{HFOV} / 2)}{\text{Image Width}}$$

---

## 4. Camera Motion Compensation (CMC)
Drone hovering induces small platform vibrations and wind drift. Sentinel's `CameraMotionCompensator` tracks sparse background features using Lucas-Kanade optical flow and RANSAC affine estimation, ensuring that crowd motion vectors reflect actual human movement rather than drone drift.


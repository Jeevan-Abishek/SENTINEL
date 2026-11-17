# SENTINEL — Technical Architecture Document

## System Vision
**SENTINEL** (*See • Understand • Warn*) is an aerial intelligence and crowd-risk monitoring platform that processes drone video and telemetry streams to deliver early warnings for crowd safety and public event management.

---

## 1. High-Level System Architecture

```
                          [ Drone Surveillance Platform ]
                       Video (RTSP / MP4) + Telemetry Link
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SENTINEL CORE ENGINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. INGESTION & SYNCHRONIZATION                                              │
│    ├── DroneStreamAdapter (RTSP reconnect, file fallback)                   │
│    ├── TelemetrySynchronizer (GPS, Altitude, Gimbal orientation)            │
│    └── FramePreprocessor & QualityChecker (Blur, lighting, letterbox)       │
│                                                                             │
│ 2. EGO-MOTION COMPENSATION & GEOSPATIAL CALIBRATION                         │
│    ├── CameraMotionCompensator (Optical Flow + Affine RANSAC)                │
│    ├── GroundResolutionCalculator (GSD m/pixel from altitude & FOV)         │
│    └── CoordinateCalibrator (4+ point homography warp to ground meters)     │
│                                                                             │
│ 3. COMPUTER VISION INFERENCE                                                │
│    ├── YOLODetector (Ultralytics YOLOv8n, person class 0 filtering)          │
│    └── Multi-Object Tracker (ByteTrack with Kalman filtering & history)     │
│                                                                             │
│ 4. MULTI-ZONE CROWD ANALYTICS                                               │
│    ├── ZoneManager & PolygonZoneValidator (Shapely containment queries)     │
│    ├── PeopleCounter (Current visible, active tracks, cumulative entries)   │
│    ├── Calibrated DensityEstimator (True people/m² vs estimated occupancy)  │
│    ├── MovementAnalyzer (Velocity vectors, dominant direction, sudden turns)│
│    ├── FlowAnalyzer (Virtual crossing lines, ingress/egress net rates)      │
│    └── BottleneckDetector (Compression index, persistent flow drops)        │
│                                                                             │
│ 5. EXPLAINABLE RISK ENGINE                                                  │
│    ├── Dynamic Multi-Factor Weighted Scoring                                │
│    │     Risk = 0.30·Density + 0.20·Occupancy + 0.20·Movement +             │
│    │            0.15·FlowImbalance + 0.15·Bottleneck                        │
│    ├── Exponential Temporal Smoothing (prevents jitter)                     │
│    ├── Threshold Classification (SAFE < 30 ≤ WARNING < 60 ≤ CRITICAL)       │
│    └── AlertManager (Cooldown deduplication, root-cause explanation)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        [ FastAPI REST & WebSocket ]            [ Streamlit Dashboard ]
        • POST /api/v1/video/upload             • Live Annotated Video Feed
        • GET /api/v1/events & /zones           • Real-time Folium Map & Zones
        • GET /api/v1/alerts                    • Calibrated Risk Gauges
        • WS /ws (Real-time telemetry)          • Human Operator Acknowledge
```

---

## 2. Key Modules and Responsibilities

### A. Video Ingestion & Telemetry (`app/video/`)
- `adapter.py`: Abstract `VideoAdapter` with implementations for `FileVideoAdapter`, `RTSPVideoAdapter` (with exponential backoff reconnect), and `MockVideoAdapter`.
- `validator.py`: Video file header inspection, codec validation, and duration/fps/frame count verification.
- `telemetry.py`: Data classes for `GPSCoordinate`, `GimbalOrientation`, and `DroneTelemetry`. Explicit `is_simulated` flag ensures operators always know whether telemetry is real or synthetic.
- `camera_motion.py`: Lucas-Kanade optical flow feature tracking with RANSAC-based affine estimation to compensate for drone vibration and hover drift.
- `drone_adapter.py`: Bundles video frames, synchronized telemetry, and ego-motion matrices into `DroneFrameBundle`.

### B. Detection & Tracking (`app/detection/`, `app/tracking/`)
- `yolo_detector.py`: Person-only filtering (`class_id == 0`), confidence thresholding, NMS IoU pruning.
- `bytetrack.py` & `tracker.py`: Two-stage data association preserving track continuity during brief occlusions. Generates anonymous, non-PII track identifiers.

### C. Geospatial Calibration & Zones (`app/mapping/`)
- `calibration.py`: Computes 3x3 homography matrix $H$ mapping camera pixel coordinates $(u, v)$ to metric ground plane coordinates $(X, Y)$:
  $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
- `GroundResolutionCalculator`: Computes ground footprint from altitude $h$ and field of view $\theta$:
  $$\text{Footprint Width} = 2 \cdot h \cdot \tan\left(\frac{\text{HFOV}}{2}\right)$$
- `zones.py`: Polygon boundary validation using Shapely. Calculates ground area in $m^2$ and true physical density ($\text{people}/m^2$).

### D. Explainable Risk Engine (`app/risk/`)
- Evaluates density, occupancy, velocity variance, flow imbalance, and bottleneck persistence.
- Exponential temporal smoothing:
  $$S_t = \alpha \cdot R_t + (1 - \alpha) \cdot S_{t-1}$$
- Classifies risk levels into `SAFE`, `WARNING`, and `CRITICAL`.
- Generates human-understandable alert explanations detailing specific contributing factors.

---

## 3. Responsible AI & Public Safety Boundaries
1. **No Predictive Determinism**: The system identifies measurable, observable risk indicators; it does not claim to predict stampedes with certainty.
2. **Anonymous Tracking Only**: Zero facial recognition or biometric identification. Anonymous track integers only.
3. **Mandatory Human Verification**: Critical alerts require human operator acknowledgment before automated operational escalation.
4. **No Drone Flight Control**: Passive surveillance sensor only; zero control over drone flight dynamics or emergency dispatch.


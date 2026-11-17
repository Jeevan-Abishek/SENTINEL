# SENTINEL — See • Understand • Warn
### AI-Powered Drone-Based Real-Time Crowd Risk Detection & Monitoring Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-210%20Passed-brightgreen.svg)](sentinel/tests)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/CV-Ultralytics%20YOLOv8-FF6F00.svg)](https://github.com/ultralytics/ultralytics)

---

## 1. Overview

**SENTINEL** is an aerial intelligence platform designed for public safety officials, event organizers, and drone surveillance operators. By ingesting video streams from commercial drones, Sentinel extracts crowd metrics, detects bottlenecks and dangerous accumulation patterns, computes explainable risk scores, and generates early warnings to help human decision-makers intervene before critical crowd incidents unfold.

> **Responsible AI Statement:** Sentinel identifies measurable, empirical risk indicators (such as density surges, flow constriction, and localized bottlenecks) to support human-supervised decision-making. It does not claim to predict stampedes with certainty, does not perform facial recognition, uses strictly anonymous track IDs, and does not autonomously dispatch emergency services.

---

## 2. Core Capabilities

1. **Video & Stream Ingestion**: Recorded video (MP4, AVI, MOV), RTSP live streams with automatic reconnection and exponential backoff, and synthetic test streams.
2. **Real YOLOv8 Person Detection**: Configurable model path (`yolov8n.pt`), confidence thresholds, and person class filtering (`class_id == 0`).
3. **Multi-Object Tracking**: ByteTrack implementation with Kalman filtering, trajectory persistence, and anonymous tracking IDs.
4. **Camera Motion Compensation (CMC)**: Lucas-Kanade optical flow feature tracking with RANSAC affine estimation to remove drone ego-motion and drift.
5. **Crowd Analytics**:
   - People counting (visible, active tracked, zone counts).
   - Calibrated crowd density ($\text{people}/m^2$) based on ground plane homography or altitude-based ground sampling distance (GSD).
   - Movement direction rose plots and sudden turn detection.
   - Virtual tripwire ingress/egress flow rates.
   - Bottleneck detection based on compression indices and persistent flow drops.
6. **Explainable Risk Engine**:
   - Multi-factor weighted risk score ($\text{Density} \times 0.30 + \text{Occupancy} \times 0.20 + \text{Movement} \times 0.20 + \text{Flow} \times 0.15 + \text{Bottleneck} \times 0.15$).
   - Temporal exponential smoothing preventing threshold flickering.
   - Classified risk states: `SAFE`, `WARNING`, `CRITICAL`.
7. **Early Warning & Human Operator Verification**:
   - Alert cooldown deduplication.
   - Root-cause explanations for each warning.
   - Mandatory human operator acknowledgment workflow.
8. **Real-Time Monitoring Dashboard**: Streamlit-based interactive operations console with annotated video scrubber, Folium geospatial venue map, zone risk breakdowns, and live alert cards.
9. **Full REST & WebSocket API**: FastAPI backend for integration into central emergency command and dispatch platforms.

---

## 3. Project Structure

```
sentinel/
├── app/
│   ├── config.py                   # Central pydantic settings & env management
│   ├── video/                      # Ingestion, RTSP, CMC, telemetry sync, validator
│   ├── detection/                  # Ultralytics YOLO detector & models
│   ├── tracking/                   # ByteTrack multi-object tracker & Kalman filters
│   ├── analytics/                  # Counting, density, movement, flow, bottleneck
│   ├── risk/                       # Risk engine, weighted scoring, alert rules
│   ├── mapping/                    # Homography calibration, GSD, Shapely zones, Folium
│   ├── database/                   # SQLAlchemy async models & CRUD repositories
│   ├── pipeline/                   # Asynchronous multi-stage stream processor
│   └── api/                        # FastAPI REST routes & WebSocket broadcaster
├── dashboard/
│   └── app.py                      # Real-time Streamlit monitoring dashboard
├── data/
│   ├── models/yolov8n.pt           # Real YOLOv8n weights file
│   └── sample_videos/              # Recorded drone test videos
├── scripts/
│   ├── run_video_demo.py           # End-to-end video pipeline demonstration
│   └── benchmark_performance.py    # Latency & throughput benchmarking script
├── tests/
│   ├── unit/                       # 208 comprehensive unit tests
│   └── integration/                # Full end-to-end video pipeline tests
├── Dockerfile                      # Production Docker container definition
├── docker-compose.yml              # Multi-container backend & dashboard orchestration
├── requirements.txt                # Production dependency specification
├── ARCHITECTURE.md                 # Detailed architecture documentation
├── TESTING.md                      # Complete testing and verification guide
└── DRONE_INTEGRATION.md            # Hardware & stream integration guide
```

---

## 4. Quick Start Guide

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### Installation
```bash
# 1. Clone/Navigate to workspace
cd sentinel

# 2. Configure environment
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline Demo
Process the included drone crowd video through the entire detection, tracking, analytics, and risk pipeline:
```bash
python scripts/run_video_demo.py --frames 30
```

### Running the Latency Benchmark
Measure exact stage-by-stage latencies:
```bash
python scripts/benchmark_performance.py
```

### Launching Backend API
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API docs available at `http://localhost:8000/docs`.

### Launching Streamlit Operator Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

## 5. Running the Test Suite

Execute all 210 automated unit and integration tests:
```bash
pytest --capture=sys -v
```

---

## 6. Docker Deployment

Launch both the backend API and Streamlit dashboard using Docker Compose:
```bash
docker-compose up --build -d
```
- API: `http://localhost:8000/health`
- Dashboard: `http://localhost:8501`

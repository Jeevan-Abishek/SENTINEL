# SENTINEL

AI-Powered Drone-Based Real-Time Crowd Risk Detection System

## Overview

SENTINEL is a modular, scalable software platform for crowd monitoring and early-warning during large public gatherings. It processes aerial video from drone surveillance cameras to detect people, track crowd movement, analyze density and flow patterns, and generate risk assessments and alerts for safety operators.

## Features

- **Video Input**: Recorded video files, RTSP streams, mock adapter for development
- **Person Detection**: YOLOv8/v11 person detection optimized for aerial footage
- **Tracking**: ByteTrack/BoT-SORT multi-object tracking with validation
- **Re-Identification**: Optional appearance-based Re-ID for track recovery
- **Crowd Analytics**: Density, occupancy, movement, flow, and bottleneck analysis
- **Risk Engine**: Configurable weighted risk scoring with temporal smoothing
- **Alert System**: Real-time alerts with acknowledgment and resolution workflow
- **Dashboard**: Streamlit-based monitoring dashboard for safety operators
- **Mapping**: Interactive map with zone management and GPS calibration
- **API**: FastAPI backend with WebSocket for real-time updates

## Architecture

```
EVENT VENUE → GPS/MAP → ZONE CALIBRATION → DRONE CAMERA → VIDEO STREAM
    ↓
FRAME PREPROCESSING → YOLO DETECTION → TRACKING & VALIDATION → RE-ID
    ↓
PEOPLE COUNTING → CROWD ANALYTICS (Density, Occupancy, Movement, Flow, Bottleneck)
    ↓
RISK ENGINE → ALERTS → STREAMLIT DASHBOARD → SAFETY OPERATOR
```

## Requirements

- Python 3.11+
- PostgreSQL 16+ with PostGIS
- Redis 7+
- FFmpeg
- NVIDIA GPU (optional, for CUDA acceleration)

## Quick Start

### 1. Clone and Setup

```bash
cd sentinel
cp .env.example .env
# Edit .env with your configuration
```

### 2. Using Docker (Recommended)

```bash
cd docker
docker-compose up -d
```

This starts:
- PostgreSQL with PostGIS on port 5432
- Redis on port 6379
- API server on port 8000
- Dashboard on port 8501

### 3. Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Set up database
cp .env.example .env
# Edit .env with local database URL

# Run database migrations
alembic upgrade head

# Start API server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start dashboard
streamlit run dashboard/app.py --server.port 8501
```

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL async connection URL | Required |
| `YOLO_MODEL_PATH` | Path to YOLO model weights | `models/yolov8n.pt` |
| `TRACKER_TYPE` | Tracker: `bytetrack` or `botsort` | `bytetrack` |
| `DEFAULT_CONFIDENCE_THRESHOLD` | Detection confidence threshold | `0.5` |
| `DRONE_MOCK_MODE` | Use mock adapter for development | `true` |
| `RISK_SAFE_THRESHOLD` | Risk score for SAFE level | `30` |
| `RISK_WARNING_THRESHOLD` | Risk score for WARNING level | `60` |
| `RISK_CRITICAL_THRESHOLD` | Risk score for CRITICAL level | `80` |

## Project Structure

```
sentinel/
├── app/                    # Backend application
│   ├── config.py          # Configuration management
│   ├── database/          # Database models and session
│   ├── video/             # Video input and processing
│   ├── detection/         # YOLO person detection
│   ├── tracking/          # Tracking and validation
│   ├── analytics/         # Crowd analytics modules
│   ├── risk/              # Risk scoring engine
│   ├── alerts/            # Alert management
│   ├── mapping/           # GPS mapping and calibration
│   ├── api/               # FastAPI routes and WebSocket
│   └── utils/             # Utilities
├── dashboard/             # Streamlit dashboard
│   ├── pages/             # Dashboard pages
│   ├── components/        # Reusable components
│   └── services/          # API/WebSocket clients
├── tests/                 # Test suite
├── data/                  # Data files
│   ├── sample_videos/     # Test videos
│   └── models/            # Model weights
├── docker/                # Docker configuration
├── scripts/               # Utility scripts
└── requirements*.txt      # Python dependencies
```

## Development Phases

### Phase 1: Project Setup ✓
- Folder structure, dependencies, configuration, video input, database models

### Phase 2: Detection (In Progress)
- Frame preprocessing, YOLO detection, visualization

### Phase 3: Tracking
- ByteTrack/BoT-SORT integration, validation, Re-ID

### Phase 4: Counting & Analytics
- People counting, zone-wise analytics

### Phase 5: Advanced Analytics
- Density, occupancy, movement, flow, bottleneck detection

### Phase 6: Risk & Alerts
- Risk engine, alert system, notifications

### Phase 7: Backend API
- FastAPI routes, WebSocket, database integration

### Phase 8: Dashboard
- All 13 dashboard pages, real-time updates

### Phase 9: Mapping
- GPS calibration, zone management, interactive map

### Phase 10: Live Integration
- RTSP support, drone telemetry, end-to-end testing

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=dashboard

# Run specific test module
pytest tests/unit/test_video_processing.py -v
```

## Model Weights

Download YOLO models:
```bash
# YOLOv8 nano (default, fast)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P data/models/

# YOLOv8 small (better accuracy)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -P data/models/

# YOLOv11 nano (latest)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -P data/models/
```

For aerial-specific detection, consider fine-tuning on datasets like:
- VisDrone
- UAVDT
- DroneCrowd

## API Documentation

When the API server is running:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Dashboard

Access the monitoring dashboard at: http://localhost:8501

## License

Proprietary - All rights reserved.

## Safety Disclaimer

SENTINEL detects measurable crowd-risk indicators and generates warnings. It does NOT predict stampedes or guarantee safety outcomes. Human operator confirmation is required for all safety actions. The system is a decision-support tool, not an autonomous emergency response system.
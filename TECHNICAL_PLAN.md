# SENTINEL - Technical Implementation Plan

## Project Overview
AI-Powered Drone-Based Real-Time Crowd Risk Detection System

## Folder Structure
```
sentinel/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py               # Database base and session
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── event.py          # Event venue management
│   │   │   ├── zone.py           # Zone management
│   │   │   ├── drone.py          # Drone management
│   │   │   ├── detection.py      # Detection records
│   │   │   ├── track.py          # Tracking records
│   │   │   ├── analytics.py      # Analytics records
│   │   │   ├── risk.py           # Risk records
│   │   │   ├── alert.py          # Alert records
│   │   │   └── operator.py       # Operator records
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── event_repo.py
│   │       ├── zone_repo.py
│   │       └── ...
│   ├── video/
│   │   ├── __init__.py
│   │   ├── adapter.py            # Video adapter base class
│   │   ├── file_adapter.py       # Local file adapter
│   │   ├── rtsp_adapter.py       # RTSP stream adapter
│   │   ├── mock_adapter.py       # Mock adapter for development
│   │   ├── processor.py          # Frame processing pipeline
│   │   └── telemetry.py          # Telemetry data handling
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py      # YOLO person detection
│   │   ├── models.py             # Detection data models
│   │   └── visualization.py      # Bounding box visualization
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py            # ByteTrack/BoT-SORT integration
│   │   ├── validator.py          # Second-stage validation
│   │   ├── reid.py               # Appearance Re-ID (optional)
│   │   └── models.py             # Tracking data models
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── counting.py           # People counting
│   │   ├── density.py            # Density analysis
│   │   ├── occupancy.py          # Occupancy analysis
│   │   ├── movement.py           # Movement analysis
│   │   ├── flow.py               # Flow analysis
│   │   ├── bottleneck.py         # Bottleneck detection
│   │   └── models.py             # Analytics data models
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── engine.py             # Risk scoring engine
│   │   ├── models.py             # Risk data models
│   │   └── rules.py              # Configurable risk rules
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── manager.py            # Alert management
│   │   ├── models.py             # Alert data models
│   │   └── notifications.py      # Notification handlers
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── calibration.py        # Coordinate calibration
│   │   ├── zones.py              # Zone management
│   │   └── map_renderer.py       # Map visualization
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── events.py
│   │   │   ├── drones.py
│   │   │   ├── zones.py
│   │   │   ├── detections.py
│   │   │   ├── tracking.py
│   │   │   ├── analytics.py
│   │   │   ├── risk.py
│   │   │   ├── alerts.py
│   │   │   └── reports.py
│   │   ├── websocket.py          # WebSocket for real-time updates
│   │   └── dependencies.py       # FastAPI dependencies
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── exceptions.py
│       └── helpers.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py                    # Streamlit main app
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── overview.py
│   │   ├── event_management.py
│   │   ├── live_monitoring.py
│   │   ├── digital_map.py
│   │   ├── zone_management.py
│   │   ├── person_detection.py
│   │   ├── crowd_tracking.py
│   │   ├── crowd_analytics.py
│   │   ├── density_heatmap.py
│   │   ├── risk_monitoring.py
│   │   ├── alert_center.py
│   │   ├── historical_reports.py
│   │   └── system_settings.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── widgets.py
│   │   ├── charts.py
│   │   └── map.py
│   └── services/
│       ├── __init__.py
│       ├── api_client.py
│       └── websocket_client.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_video_processing.py
│   │   ├── test_detection.py
│   │   ├── test_tracking.py
│   │   ├── test_analytics.py
│   │   ├── test_risk_engine.py
│   │   └── test_alerts.py
│   └── integration/
│       ├── test_api.py
│       └── test_pipeline.py
├── data/
│   ├── sample_videos/            # Sample drone videos for testing
│   └── models/                   # Model weights
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── scripts/
│   ├── init_db.py
│   ├── run_detection.py
│   └── download_models.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
└── TECHNICAL_PLAN.md
```

## Phase 1: Project Setup (Current)
- [x] Create folder structure
- [ ] Create requirements.txt with all dependencies
- [ ] Create .env.example configuration template
- [ ] Create Docker and Docker Compose configuration
- [ ] Create core configuration module (app/config.py)
- [ ] Create video input module with recorded video support
- [ ] Create basic database models
- [ ] Create README.md with setup instructions
- [ ] Run initial tests to verify setup

## Phase 2: Frame Preprocessing and YOLO Detection
- Video frame extraction and preprocessing
- YOLO person detection module
- Detection visualization
- Unit tests for detection

## Phase 3: Tracking and Validation
- ByteTrack/BoT-SORT integration
- Second-stage scanning/validation
- Optional Re-ID module
- Tracking tests

## Phase 4: People Counting and Crowd Analytics
- Counting module
- Zone-wise counting
- Entry/exit counting
- Analytics tests

## Phase 5: Advanced Analytics
- Density analysis
- Occupancy analysis
- Movement analysis
- Flow analysis
- Bottleneck detection
- Analytics tests

## Phase 6: Risk Engine and Alerts
- Risk scoring engine
- Configurable rules
- Alert management
- Notification system
- Risk/alert tests

## Phase 7: FastAPI Backend and Database
- API routes for all modules
- WebSocket for real-time updates
- Database migrations
- API tests

## Phase 8: Streamlit Dashboard
- All 13 pages
- Real-time updates via WebSocket
- Interactive map
- Dashboard tests

## Phase 9: GPS Mapping and Zone Calibration
- Coordinate calibration
- Zone management
- Map visualization
- Mapping tests

## Phase 10: Live Drone Integration
- RTSP stream support
- Drone telemetry integration
- Mock adapter for development
- End-to-end testing

## Key Technical Decisions

### Video Processing
- Use OpenCV for frame extraction
- Process frames on-demand (streaming), not loading entire video
- Configurable frame skip for performance
- Support for multiple input sources via adapter pattern

### Detection
- Ultralytics YOLO (YOLOv8/v11) for person detection
- Configurable confidence threshold
- Class filtering for 'person' only
- Support for custom aerial-trained weights

### Tracking
- ByteTrack as primary (fast, robust)
- BoT-SORT as alternative (better Re-ID integration)
- Track management with unique IDs
- Handle occlusions and missed detections

### Analytics
- Ground-plane calibration when telemetry available
- Fallback to image-plane metrics when calibration unavailable
- Temporal smoothing for stability
- Zone-based analytics

### Risk Engine
- Weighted scoring (configurable)
- Zone-specific thresholds
- Temporal smoothing
- Explainable risk factors

### Database
- PostgreSQL with PostGIS for geospatial
- SQLAlchemy ORM
- Async support with asyncpg
- Proper indexing for time-series data

### Dashboard
- Streamlit for rapid development
- WebSocket for live updates
- Folium for interactive maps
- Professional UI for operators

## Dependencies (requirements.txt)
```
# Core
fastapi==0.110.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
pydantic-settings==2.3.0
python-dotenv==1.0.0

# Database
sqlalchemy==2.0.30
asyncpg==0.29.0
alembic==1.13.0
geoalchemy2==0.14.0

# Computer Vision
opencv-python-headless==4.9.0
ultralytics==8.2.0
torch==2.3.0
torchvision==0.18.0
numpy==1.26.0

# Tracking
lap==0.4.0
scipy==1.13.0

# Video Processing
ffmpeg-python==0.2.0

# Mapping
folium==0.16.0
geojson==3.1.0
shapely==2.0.0
pyproj==3.6.0

# Dashboard
streamlit==1.35.0
plotly==5.21.0
pandas==2.2.0

# WebSocket
websockets==12.0

# Testing
pytest==8.2.0
pytest-asyncio==0.23.0
pytest-cov==4.1.0
httpx==0.27.0

# Utilities
loguru==0.7.0
python-multipart==0.0.9
aiofiles==23.2.0
```

## Development Requirements (requirements-dev.txt)
```
-r requirements.txt
black==24.3.0
isort==5.13.0
flake8==7.0.0
mypy==1.10.0
pre-commit==3.7.0
```

## Environment Variables (.env.example)
```
# Application
APP_NAME=SENTINEL
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql+asyncpg://sentinel:sentinel@localhost:5432/sentinel
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501

# Video Processing
DEFAULT_CONFIDENCE_THRESHOLD=0.5
DEFAULT_IOU_THRESHOLD=0.45
MAX_DETECTIONS_PER_FRAME=1000
FRAME_SKIP=1
PROCESSING_WIDTH=640
PROCESSING_HEIGHT=640

# YOLO Model
YOLO_MODEL_PATH=models/yolov8n.pt
YOLO_DEVICE=cpu
YOLO_HALF_PRECISION=false

# Tracking
TRACKER_TYPE=bytetrack
TRACK_BUFFER=30
MATCH_THRESHOLD=0.8
MIN_BOX_AREA=10

# Re-ID (Optional)
REID_ENABLED=false
REID_MODEL_PATH=models/osnet_x1_0.pth
REID_THRESHOLD=0.6

# Analytics
DENSITY_GRID_SIZE=1.0  # meters per cell
OCCUPANCY_WARNING_THRESHOLD=0.7
OCCUPANCY_CRITICAL_THRESHOLD=0.9
FLOW_WINDOW_SECONDS=60
BOTTLENECK_PERSISTENCE_FRAMES=30

# Risk Engine
RISK_DENSITY_WEIGHT=0.3
RISK_OCCUPANCY_WEIGHT=0.2
RISK_MOVEMENT_WEIGHT=0.2
RISK_FLOW_WEIGHT=0.15
RISK_BOTTLENECK_WEIGHT=0.15
RISK_SAFE_THRESHOLD=30
RISK_WARNING_THRESHOLD=60
RISK_CRITICAL_THRESHOLD=80
RISK_TEMPORAL_SMOOTHING=0.3

# Alerts
ALERT_COOLDOWN_SECONDS=30
ALERT_MAX_ACTIVE=100
ALERT_SOUND_ENABLED=true

# Drone (Mock for development)
DRONE_MOCK_MODE=true
DRONE_RTSP_URL=
DRONE_TELEMETRY_PORT=14550

# Mapping
DEFAULT_MAP_CENTER_LAT=0.0
DEFAULT_MAP_CENTER_LON=0.0
DEFAULT_MAP_ZOOM=18
```

## Docker Configuration

### Dockerfile (Backend)
- Base: python:3.11-slim
- Install system dependencies (ffmpeg, libgl1, etc.)
- Copy requirements and install
- Copy application code
- Run with uvicorn

### Dockerfile.dashboard (Frontend)
- Base: python:3.11-slim
- Install requirements
- Copy dashboard code
- Run with streamlit

### docker-compose.yml
- Services: postgres, redis, api, dashboard
- Volumes for data persistence
- Network configuration
- Health checks
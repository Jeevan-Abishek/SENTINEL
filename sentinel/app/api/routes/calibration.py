from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
import json


router = APIRouter(prefix="/calibration", tags=["Calibration"])


class CalibrationPointRequest(BaseModel):
    image_x: float = Field(..., description="X coordinate in image (pixels)")
    image_y: float = Field(..., description="Y coordinate in image (pixels)")
    world_x: float = Field(..., description="X coordinate in world (meters or lat)")
    world_y: float = Field(..., description="Y coordinate in world (meters or lon)")


class CalibrationPointResponse(BaseModel):
    image_x: float
    image_y: float
    world_x: float
    world_y: float


class CalibrationResultResponse(BaseModel):
    homography_matrix: Optional[List[List[float]]] = None
    reprojection_error: float
    is_valid: bool
    world_bounds: Optional[List[float]] = None


class CalibrationDataRequest(BaseModel):
    points: List[CalibrationPointRequest]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class CalibrationDataResponse(BaseModel):
    points: List[CalibrationPointResponse]
    homography: Optional[List[List[float]]] = None
    world_bounds: Optional[List[float]] = None


# In-memory calibration storage (replace with database in production)
_calibrations: Dict[str, Dict[str, Any]] = {}


@router.post("/{event_id}/points", response_model=CalibrationPointResponse, status_code=status.HTTP_201_CREATED)
async def add_calibration_point(
    event_id: str,
    point: CalibrationPointRequest,
):
    if event_id not in _calibrations:
        _calibrations[event_id] = {"points": []}
    
    _calibrations[event_id]["points"].append(point.model_dump())
    return point


@router.get("/{event_id}/points", response_model=List[CalibrationPointResponse])
async def get_calibration_points(event_id: str):
    if event_id not in _calibrations:
        return []
    return _calibrations[event_id].get("points", [])


@router.delete("/{event_id}/points/{point_index}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_calibration_point(event_id: str, point_index: int):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    points = _calibrations[event_id].get("points", [])
    if point_index < 0 or point_index >= len(points):
        raise HTTPException(status_code=404, detail="Point index out of range")
    
    points.pop(point_index)
    _calibrations[event_id]["homography"] = None


@router.post("/{event_id}/compute", response_model=CalibrationResultResponse)
async def compute_homography(event_id: str):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    points = _calibrations[event_id].get("points", [])
    if len(points) < 4:
        raise HTTPException(status_code=400, detail="At least 4 calibration points required")
    
    import numpy as np
    import cv2
    
    image_points = np.array([[p["image_x"], p["image_y"]] for p in points], dtype=np.float32)
    world_points = np.array([[p["world_x"], p["world_y"]] for p in points], dtype=np.float32)
    
    H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
    
    if H is None:
        raise HTTPException(status_code=500, detail="Failed to compute homography")
    
    # Compute reprojection error
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_points, axis=1)
    mean_error = float(np.mean(errors))
    
    # Compute world bounds
    min_x = float(np.min(world_points[:, 0]))
    max_x = float(np.max(world_points[:, 0]))
    min_y = float(np.min(world_points[:, 1]))
    max_y = float(np.max(world_points[:, 1]))
    
    result = {
        "homography_matrix": H.tolist(),
        "reprojection_error": mean_error,
        "is_valid": mean_error < 10.0,
        "world_bounds": [min_x, min_y, max_x, max_y],
    }
    
    _calibrations[event_id]["homography"] = H.tolist()
    _calibrations[event_id]["world_bounds"] = [min_x, min_y, max_x, max_y]
    
    return result


@router.get("/{event_id}/result", response_model=CalibrationResultResponse)
async def get_calibration_result(event_id: str):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    calib = _calibrations[event_id]
    if "homography" not in calib or calib["homography"] is None:
        raise HTTPException(status_code=404, detail="Homography not computed yet")
    
    return {
        "homography_matrix": calib["homography"],
        "reprojection_error": calib.get("reprojection_error", 0.0),
        "is_valid": calib.get("is_valid", True),
        "world_bounds": calib.get("world_bounds"),
    }


@router.post("/{event_id}/load", status_code=status.HTTP_200_OK)
async def load_calibration(event_id: str, data: CalibrationDataRequest):
    import numpy as np
    import cv2
    
    points = [p.model_dump() for p in data.points]
    _calibrations[event_id] = {"points": points}
    
    if len(points) >= 4:
        image_points = np.array([[p["image_x"], p["image_y"]] for p in points], dtype=np.float32)
        world_points = np.array([[p["world_x"], p["world_y"]] for p in points], dtype=np.float32)
        
        H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
        
        if H is not None:
            projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
            errors = np.linalg.norm(projected - world_points, axis=1)
            mean_error = float(np.mean(errors))
            
            min_x = float(np.min(world_points[:, 0]))
            max_x = float(np.max(world_points[:, 0]))
            min_y = float(np.min(world_points[:, 1]))
            max_y = float(np.max(world_points[:, 1]))
            
            _calibrations[event_id]["homography"] = H.tolist()
            _calibrations[event_id]["reprojection_error"] = mean_error
            _calibrations[event_id]["is_valid"] = mean_error < 10.0
            _calibrations[event_id]["world_bounds"] = [min_x, min_y, max_x, max_y]
    
    return {"status": "loaded", "point_count": len(points)}


@router.get("/{event_id}/export", response_model=CalibrationDataResponse)
async def export_calibration(event_id: str):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    calib = _calibrations[event_id]
    return {
        "points": calib.get("points", []),
        "homography": calib.get("homography"),
        "world_bounds": calib.get("world_bounds"),
    }


@router.post("/{event_id}/transform", response_model=Dict[str, float])
async def transform_point(
    event_id: str,
    image_x: float = Query(..., description="X coordinate in image"),
    image_y: float = Query(..., description="Y coordinate in image"),
):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    calib = _calibrations[event_id]
    if "homography" not in calib or calib["homography"] is None:
        raise HTTPException(status_code=400, detail="Homography not computed yet")
    
    import numpy as np
    import cv2
    
    H = np.array(calib["homography"], dtype=np.float32)
    point = np.array([[[image_x, image_y]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, H)
    
    return {
        "world_x": float(world_point[0, 0, 0]),
        "world_y": float(world_point[0, 0, 1]),
    }


@router.post("/{event_id}/inverse_transform", response_model=Dict[str, float])
async def inverse_transform_point(
    event_id: str,
    world_x: float = Query(..., description="X coordinate in world"),
    world_y: float = Query(..., description="Y coordinate in world"),
):
    if event_id not in _calibrations:
        raise HTTPException(status_code=404, detail="Calibration not found")
    
    calib = _calibrations[event_id]
    if "homography" not in calib or calib["homography"] is None:
        raise HTTPException(status_code=400, detail="Homography not computed yet")
    
    import numpy as np
    import cv2
    
    H = np.array(calib["homography"], dtype=np.float32)
    H_inv = np.linalg.inv(H)
    point = np.array([[[world_x, world_y]]], dtype=np.float32)
    image_point = cv2.perspectiveTransform(point, H_inv)
    
    return {
        "image_x": float(image_point[0, 0, 0]),
        "image_y": float(image_point[0, 0, 1]),
    }
"""
Camera Motion Compensation Module.
Estimates pan, tilt, zoom, and translation between video frames using Lucas-Kanade optical flow.
"""

from app.video.camera_motion import CameraMotionCompensator, CameraMotionResult

__all__ = ["CameraMotionCompensator", "CameraMotionResult"]


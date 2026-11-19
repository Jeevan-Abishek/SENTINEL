import cv2
import numpy as np
from typing import Optional, List, Tuple
import colorsys

from app.detection.models import Detection, DetectionResult


class ColorPalette:
    def __init__(self):
        self._colors = {}
        self._base_hues = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self._colors:
            hue = self._base_hues[track_id % len(self._base_hues)]
            saturation = 0.9
            value = 0.95
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self._colors[track_id] = tuple(int(c * 255) for c in rgb)
        return self._colors[track_id]

    def get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        hues = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        hue = hues[class_id % len(hues)]
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)


PALETTE = ColorPalette()


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
    show_confidence: bool = True,
    show_class: bool = True,
    show_track_id: bool = True,
    show_center: bool = False,
    show_foot: bool = False,
    font_scale: float = 0.6,
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, detection.bbox)

    if color is None:
        if detection.track_id is not None:
            color = PALETTE.get_color(detection.track_id)
        else:
            color = PALETTE.get_class_color(detection.class_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Optional centroid point marker
    if show_center:
        cx, cy = int(detection.center_x), int(detection.center_y)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

    # Optional foot / ground-contact point marker
    if show_foot:
        fx, fy = int(detection.center_x), int(detection.y2)
        cv2.circle(frame, (fx, fy), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 2, (0, 140, 255), -1, cv2.LINE_AA)

    label_parts = []
    if show_class:
        label_parts.append(detection.class_name)
    if show_confidence:
        label_parts.append(f"{detection.confidence:.2f}")
    if show_track_id and detection.track_id is not None:
        label_parts.append(f"ID:{detection.track_id}")

    if label_parts:
        label = " ".join(label_parts)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w + 5, y1),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    color: Optional[Tuple[int, int, int]] = None,
    show_center: bool = False,
    show_foot: bool = False,
    **kwargs,
) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        draw_detection(annotated, det, color=color, show_center=show_center, show_foot=show_foot, **kwargs)
    return annotated


def draw_detection_result(
    frame: np.ndarray,
    result: DetectionResult,
    show_center: bool = False,
    show_foot: bool = False,
    **kwargs,
) -> np.ndarray:
    return draw_detections(frame, result.detections, show_center=show_center, show_foot=show_foot, **kwargs)


def draw_count_overlay(
    frame: np.ndarray,
    count: int,
    position: Tuple[int, int] = (20, 40),
    font_scale: float = 1.0,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    text = f"People: {count}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - baseline - 5),
        (x + text_w + 10, y + baseline + 5),
        background_color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_fps_overlay(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (20, 80),
    font_scale: float = 0.7,
    thickness: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    text = f"FPS: {fps:.1f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - baseline - 5),
        (x + text_w + 10, y + baseline + 5),
        background_color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_processing_info(
    frame: np.ndarray,
    result: DetectionResult,
    position: Tuple[int, int] = (20, 120),
    font_scale: float = 0.6,
    thickness: int = 1,
    color: Tuple[int, int, int] = (200, 200, 200),
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    texts = [
        f"Frame: {result.frame_number}",
        f"Time: {result.timestamp:.2f}s",
        f"Inference: {result.processing_time_ms:.1f}ms",
        f"Model: {result.model_name}",
    ]

    y = position[1]
    for text in texts:
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        x = position[0]

        cv2.rectangle(
            frame,
            (x - 3, y - text_h - baseline - 3),
            (x + text_w + 6, y + baseline + 3),
            background_color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += text_h + baseline + 8

    return frame


def create_detection_summary_image(
    frame: np.ndarray,
    result: DetectionResult,
    show_fps: float = 0.0,
) -> np.ndarray:
    annotated = frame.copy()

    annotated = draw_detection_result(annotated, result)
    annotated = draw_count_overlay(annotated, result.person_count)

    if show_fps > 0:
        annotated = draw_fps_overlay(annotated, show_fps)

    annotated = draw_processing_info(annotated, result)

    return annotated
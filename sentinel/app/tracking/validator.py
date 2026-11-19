"""
SENTINEL Second-Stage Scanning & Validation Module
Filters false positives, stray reflections, tiny noise boxes, and motion outliers.
Ensures downstream analytics (Crowd Analytics & Risk Engine) receive verified, validated tracks.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field

from app.tracking.models import TrackedPerson, TrackerConfig


class ValidationConfig(BaseModel):
    """Configuration for second-stage track candidate validation."""
    min_hits: int = Field(default=3, ge=1, description="Minimum consecutive hits for temporal confirmation")
    min_box_area: float = Field(default=10.0, ge=0.0, description="Minimum box area in px^2 to reject noise/dust")
    max_box_area: float = Field(default=500000.0, description="Maximum box area in px^2 to reject glare/artifacts")
    min_aspect_ratio: float = Field(default=0.15, description="Minimum w/h aspect ratio for aerial person view")
    max_aspect_ratio: float = Field(default=3.5, description="Maximum w/h aspect ratio for aerial person view")
    max_velocity_jump: float = Field(default=150.0, description="Maximum centroid displacement per frame (px)")
    min_confidence: float = Field(default=0.30, ge=0.0, le=1.0, description="Minimum confidence for validated track")
    enabled: bool = Field(default=True, description="Enable or disable second-stage validation filter")


class ValidationReport(BaseModel):
    """Structured report produced by second-stage track candidate scanning."""
    validated_tracks: List[TrackedPerson] = Field(default_factory=list)
    rejected_tracks: List[TrackedPerson] = Field(default_factory=list)
    rejected_count: int = 0
    validation_reasons: Dict[int, str] = Field(default_factory=dict)
    rejection_breakdown: Dict[str, int] = Field(default_factory=dict)
    total_input: int = 0
    validated_count: int = 0
    validation_rate: float = 1.0


class SecondStageValidator:
    """
    Second-Stage Scanner validating YOLO person track candidates.
    Applies temporal confirmation, bounding box geometric sanity checks,
    and motion outlier gating for aerial drone perspectives.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate_track(self, track: TrackedPerson) -> Tuple[bool, str]:
        """
        Validates an individual track against drone surveillance criteria.
        Returns (is_valid: bool, reason: str).
        """
        if not self.config.enabled:
            return True, "validation_disabled"

        # 1. Box Area Sanity Check (filter tiny noise speckles / huge glares)
        area = track.area
        if area < self.config.min_box_area:
            return False, f"min_box_area ({area:.1f} < {self.config.min_box_area})"
        if area > self.config.max_box_area:
            return False, f"max_box_area ({area:.1f} > {self.config.max_box_area})"

        # 2. Aspect Ratio Sanity Check (w/h)
        w = max(0.0, track.width)
        h = max(0.0, track.height)
        if h <= 0.0 or w <= 0.0:
            return False, "invalid_box_dimensions (width or height <= 0)"
        aspect_ratio = w / h
        if aspect_ratio < self.config.min_aspect_ratio:
            return False, f"min_aspect_ratio ({aspect_ratio:.2f} < {self.config.min_aspect_ratio})"
        if aspect_ratio > self.config.max_aspect_ratio:
            return False, f"max_aspect_ratio ({aspect_ratio:.2f} > {self.config.max_aspect_ratio})"

        # 3. Minimum Hits / Temporal Confirmation
        # Requires at least min_hits detections before promoting from candidate to validated
        if track.hits < self.config.min_hits:
            return False, f"insufficient_hits ({track.hits} < {self.config.min_hits})"

        # 4. Motion Consistency / Outlier Jump Gating
        # If track has recent centroid history, check if displacement is physically plausible
        if hasattr(track, "centroid_history") and len(track.centroid_history) >= 2:
            prev = np.array(track.centroid_history[-2])
            curr = np.array(track.centroid_history[-1])
            disp = float(np.linalg.norm(curr - prev))
            if disp > self.config.max_velocity_jump:
                return False, f"velocity_jump_outlier ({disp:.1f}px > {self.config.max_velocity_jump}px)"

        # 5. Confidence sanity check
        if track.confidence < self.config.min_confidence:
            return False, f"low_confidence ({track.confidence:.2f} < {self.config.min_confidence})"

        return True, "validated"

    def validate_tracks(self, tracks: List[TrackedPerson]) -> ValidationReport:
        """
        Validates a list of tracks, updating each track's `is_validated` and `validation_reason`.
        """
        validated = []
        rejected = []
        reasons = {}
        breakdown = {}

        for track in tracks:
            # Only validate active tracks; lost/removed tracks remain unvalidated
            if track.state != "active":
                track.is_validated = False
                track.validation_reason = f"inactive_state ({track.state})"
                continue

            is_valid, reason = self.validate_track(track)
            track.is_validated = is_valid
            track.validation_reason = reason

            if is_valid:
                validated.append(track)
            else:
                rejected.append(track)
                reasons[track.track_id] = reason
                rule_name = reason.split(" ")[0]
                breakdown[rule_name] = breakdown.get(rule_name, 0) + 1

        total = len(validated) + len(rejected)
        val_rate = len(validated) / total if total > 0 else 1.0

        return ValidationReport(
            validated_tracks=validated,
            rejected_tracks=rejected,
            rejected_count=len(rejected),
            validation_reasons=reasons,
            rejection_breakdown=breakdown,
            total_input=total,
            validated_count=len(validated),
            validation_rate=val_rate,
        )

    def filter_validated(self, tracks: List[TrackedPerson]) -> List[TrackedPerson]:
        """Convenience method returning only the validated active tracks."""
        report = self.validate_tracks(tracks)
        return report.validated_tracks


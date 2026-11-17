"""
Bottleneck Detector Module.
Identifies localized congestion, compression ratio, and flow stagnation.
"""

from app.analytics.bottleneck import BottleneckDetector, BottleneckResult, BottleneckDetectorWrapper

__all__ = ["BottleneckDetector", "BottleneckResult", "BottleneckDetectorWrapper"]


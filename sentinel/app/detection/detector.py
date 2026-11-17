from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

from app.detection.models import DetectionResult, DetectorConfig, ModelInfo


@dataclass
class DetectionContext:
    frame_number: int = 0
    timestamp: float = 0.0
    original_frame_shape: tuple[int, int] = (0, 0)


class BaseDetector(ABC):
    def __init__(self, config: DetectorConfig):
        self.config = config
        self._model = None
        self._model_info: Optional[ModelInfo] = None
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        pass

    @abstractmethod
    def detect(self, frame: "np.ndarray", context: Optional[DetectionContext] = None) -> DetectionResult:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_info(self) -> Optional[ModelInfo]:
        return self._model_info

    def validate_input(self, frame: "np.ndarray") -> bool:
        if frame is None:
            return False
        if frame.ndim != 3:
            return False
        if frame.shape[2] != 3:
            return False
        return True


class DetectionError(Exception):
    pass


class ModelLoadError(DetectionError):
    pass


class InferenceError(DetectionError):
    pass
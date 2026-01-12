"""Initialization for detectors module"""

from .base import DetectorBase, DetectorOutput
from .mediapipe_detector import MediaPipeDetector
from .yolo_detector import YOLODetector
from .mediapipe_yolo_detector import MediaPipeYoloDetector, MediaPipeYoloStrategy
from .multi_detector_engine import MultiDetectorEngine, MultiDetectorMode

# Backwards compatibility aliases
FusionDetector = MediaPipeYoloDetector
FusionStrategy = MediaPipeYoloStrategy
FusionEngine = MultiDetectorEngine
FusionMode = MultiDetectorMode

__all__ = [
    'DetectorBase',
    'DetectorOutput',
    'MediaPipeDetector',
    'YOLODetector',
    'MediaPipeYoloDetector',
    'MediaPipeYoloStrategy',
    'MultiDetectorEngine',
    'MultiDetectorMode',
    # Backwards compatibility
    'FusionDetector',
    'FusionStrategy',
    'FusionEngine',
    'FusionMode',
]

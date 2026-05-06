"""Base class for all detectors - defines consistent interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DetectorOutput:
    """Standardized output from any detector"""
    label: str                              # 'work', 'idle', 'unknown'
    confidence: float                       # 0.0-1.0
    metadata: Dict[str, Any]               # Detector-specific features
    raw_output: Optional[Any] = None       # Debug: raw detector output


class DetectorBase(ABC):
    """Abstract base class for all activity detectors"""
    
    def __init__(self, name: str, confidence_threshold: float = 0.5):
        self.name = name
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def process_frame(self, frame) -> DetectorOutput:
        """
        Process a single frame and return activity label.
        
        Args:
            frame: OpenCV frame (BGR numpy array)
        
        Returns:
            DetectorOutput with label, confidence, and metadata
        """
        pass
    
    @abstractmethod
    def close(self):
        """Cleanup resources (GPU memory, file handles, etc.)"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

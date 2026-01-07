"""Fusion detector combining YOLO (object context) + MediaPipe (motion signal)"""

import logging
from typing import Dict, Optional, List
from enum import Enum

from base import DetectorBase, DetectorOutput
from mediapipe_detector import MediaPipeDetector
from yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """How to combine signals from different detectors"""
    WEIGHTED = "weighted"        # Average with configurable weights
    CASCADE = "cascade"          # YOLO context + MediaPipe motion
    CONSENSUS = "consensus"     # Both must agree
    MAJORITY_VOTE = "majority"  # >50% confidence across detectors


class FusionDetector(DetectorBase):
    """
    Intelligently combines YOLO (object detection) and MediaPipe (motion/pose) signals.
    
    Architecture:
    - YOLO provides CONTEXT: what tools/objects are present
    - MediaPipe provides SIGNAL: hand motion and pose activity
    - Fusion combines both for robust activity detection
    
    Examples:
    1. Hand moving + tools detected = HIGH confidence WORK
    2. Hand moving + no tools = MEDIUM confidence WORK (might be gesturing)
    3. Tools present + no motion = LOW confidence WORK (idle with tools nearby)
    4. No hand motion + no tools = HIGH confidence IDLE
    """
    
    # How much weight each signal contributes (sum should = 1.0)
    DEFAULT_WEIGHTS = {
        'mediapipe_motion': 0.6,   # Motion is primary signal
        'yolo_context': 0.4,       # Context amplifies or dampens
    }
    
    def __init__(
        self,
        yolo_model: str = 'yolov8n',
        mediapipe_confidence: float = 0.5,
        yolo_confidence: float = 0.5,
        strategy: FusionStrategy = FusionStrategy.CASCADE,
        weights: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize fusion detector.
        
        Args:
            yolo_model: YOLO model name (e.g., 'yolov8n', 'yolov8s')
            mediapipe_confidence: Confidence threshold for MediaPipe
            yolo_confidence: Confidence threshold for YOLO
            strategy: How to combine signals
            weights: Optional custom weights for weighted strategy
            device: Device for YOLO (e.g., 'cuda:0', 'cpu')
        """
        super().__init__(name='fusion', confidence_threshold=0.5)
        
        self.strategy = strategy if isinstance(strategy, FusionStrategy) else FusionStrategy[strategy.upper()]
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()
        
        # Initialize sub-detectors
        self.mediapipe_detector = None
        self.yolo_detector = None
        self.frame_count = 0
        
        try:
            self.mediapipe_detector = MediaPipeDetector(
                confidence_threshold=mediapipe_confidence,
                enable_visualization=False
            )
            logger.info("MediaPipe detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MediaPipe: {e}")
        
        try:
            self.yolo_detector = YOLODetector(
                model=yolo_model,
                confidence_threshold=yolo_confidence,
                device=device
            )
            logger.info(f"YOLO detector initialized with {yolo_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO: {e}")
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = self.weights[key] / total
    
    def process_frame(self, frame) -> DetectorOutput:
        """
        Process frame using both YOLO and MediaPipe, fuse results.
        
        Returns:
            DetectorOutput with combined label and confidence
        """
        self.frame_count += 1
        
        mediapipe_result = None
        yolo_result = None
        
        # Get both signals
        if self.mediapipe_detector:
            try:
                mediapipe_result = self.mediapipe_detector.process_frame(frame)
            except Exception as e:
                logger.warning(f"MediaPipe error on frame {self.frame_count}: {e}")
        
        if self.yolo_detector:
            try:
                yolo_result = self.yolo_detector.process_frame(frame)
            except Exception as e:
                logger.warning(f"YOLO error on frame {self.frame_count}: {e}")
        
        # If both fail, return unknown
        if mediapipe_result is None and yolo_result is None:
            return DetectorOutput(
                label='unknown',
                confidence=0.0,
                metadata={'error': 'Both detectors failed'}
            )
        
        # If only one available, return it
        if mediapipe_result is None:
            return yolo_result
        if yolo_result is None:
            return mediapipe_result
        
        # Fuse both signals
        if self.strategy == FusionStrategy.CASCADE:
            return self._fuse_cascade(mediapipe_result, yolo_result)
        elif self.strategy == FusionStrategy.WEIGHTED:
            return self._fuse_weighted(mediapipe_result, yolo_result)
        elif self.strategy == FusionStrategy.CONSENSUS:
            return self._fuse_consensus(mediapipe_result, yolo_result)
        elif self.strategy == FusionStrategy.MAJORITY_VOTE:
            return self._fuse_majority(mediapipe_result, yolo_result)
        else:
            return self._fuse_weighted(mediapipe_result, yolo_result)
    
    def _fuse_cascade(self, mp_result: DetectorOutput, yolo_result: DetectorOutput) -> DetectorOutput:
        """
        CASCADE: Use YOLO context to amplify or dampen MediaPipe motion signal.
        
        Logic:
        - If MediaPipe detects hand motion (work):
          - If YOLO sees tools → CONFIRM WORK (high confidence)
          - If YOLO sees no tools → TENTATIVE WORK (medium confidence)
        - If MediaPipe detects idle:
          - If YOLO sees tools nearby → CAUTIOUS (low work confidence)
          - If YOLO sees no tools → CONFIRM IDLE (high confidence)
        """
        mp_label = mp_result.label
        yolo_label = yolo_result.label
        mp_conf = mp_result.confidence
        yolo_conf = yolo_result.confidence
        
        # Extract work indicators from YOLO metadata
        yolo_has_tools = 'work_objects' in yolo_result.metadata and len(yolo_result.metadata['work_objects']) > 0
        
        if mp_label == 'work':
            if yolo_has_tools:
                # Strong signal: motion + tools present
                combined_conf = min(1.0, mp_conf * 0.6 + yolo_conf * 0.4)
                return DetectorOutput(
                    label='work',
                    confidence=combined_conf,
                    metadata={
                        'strategy': 'cascade_confirmed',
                        'mp_signal': mp_label,
                        'yolo_context': yolo_label,
                        'mp_confidence': float(mp_conf),
                        'yolo_confidence': float(yolo_conf),
                    }
                )
            else:
                # Weaker signal: motion without tools
                combined_conf = mp_conf * 0.7  # Reduce confidence
                return DetectorOutput(
                    label='work',
                    confidence=combined_conf,
                    metadata={
                        'strategy': 'cascade_motion_only',
                        'mp_signal': mp_label,
                        'yolo_context': yolo_label,
                        'mp_confidence': float(mp_conf),
                        'yolo_confidence': float(yolo_conf),
                    }
                )
        else:  # mp_label == 'idle'
            if yolo_has_tools:
                # Tools present but no motion - might be working slowly
                combined_conf = min(1.0, (mp_conf * 0.4 + yolo_conf * 0.6))
                return DetectorOutput(
                    label='idle',  # Still call it idle but with caveats
                    confidence=combined_conf * 0.7,
                    metadata={
                        'strategy': 'cascade_tools_no_motion',
                        'mp_signal': mp_label,
                        'yolo_context': yolo_label,
                        'mp_confidence': float(mp_conf),
                        'yolo_confidence': float(yolo_conf),
                        'note': 'Tools detected but no motion',
                    }
                )
            else:
                # No motion, no tools = confirmed idle
                combined_conf = min(1.0, mp_conf * 0.5 + yolo_conf * 0.5)
                return DetectorOutput(
                    label='idle',
                    confidence=combined_conf,
                    metadata={
                        'strategy': 'cascade_confirmed_idle',
                        'mp_signal': mp_label,
                        'yolo_context': yolo_label,
                        'mp_confidence': float(mp_conf),
                        'yolo_confidence': float(yolo_conf),
                    }
                )
    
    def _fuse_weighted(self, mp_result: DetectorOutput, yolo_result: DetectorOutput) -> DetectorOutput:
        """Weighted average of both signals."""
        mp_label = mp_result.label
        yolo_label = yolo_result.label
        mp_conf = mp_result.confidence
        yolo_conf = yolo_result.confidence
        
        # Map labels to numeric values
        label_to_value = {'work': 1.0, 'idle': 0.0, 'unknown': 0.5}
        
        mp_value = label_to_value.get(mp_label, 0.5)
        yolo_value = label_to_value.get(yolo_label, 0.5)
        
        # Weighted average
        combined_value = (
            mp_value * self.weights.get('mediapipe_motion', 0.6) +
            yolo_value * self.weights.get('yolo_context', 0.4)
        )
        
        # Determine label from combined value
        if combined_value > 0.6:
            label = 'work'
        elif combined_value < 0.4:
            label = 'idle'
        else:
            label = 'unknown'
        
        # Confidence is average of both detectors' confidence
        combined_conf = (mp_conf * self.weights.get('mediapipe_motion', 0.6) + 
                        yolo_conf * self.weights.get('yolo_context', 0.4))
        
        return DetectorOutput(
            label=label,
            confidence=combined_conf,
            metadata={
                'strategy': 'weighted',
                'mp_signal': mp_label,
                'yolo_context': yolo_label,
                'mp_confidence': float(mp_conf),
                'yolo_confidence': float(yolo_conf),
                'combined_value': float(combined_value),
            }
        )
    
    def _fuse_consensus(self, mp_result: DetectorOutput, yolo_result: DetectorOutput) -> DetectorOutput:
        """Both detectors must agree on label."""
        mp_label = mp_result.label
        yolo_label = yolo_result.label
        
        if mp_label == yolo_label and mp_label != 'unknown':
            # Both agree
            combined_conf = min(1.0, (mp_result.confidence + yolo_result.confidence) / 2.0)
            return DetectorOutput(
                label=mp_label,
                confidence=combined_conf,
                metadata={
                    'strategy': 'consensus_agreed',
                    'mp_signal': mp_label,
                    'yolo_context': yolo_label,
                }
            )
        else:
            # Disagreement
            return DetectorOutput(
                label='unknown',
                confidence=0.0,
                metadata={
                    'strategy': 'consensus_disagreement',
                    'mp_signal': mp_label,
                    'yolo_context': yolo_label,
                }
            )
    
    def _fuse_majority(self, mp_result: DetectorOutput, yolo_result: DetectorOutput) -> DetectorOutput:
        """Simple majority vote."""
        mp_label = mp_result.label
        yolo_label = yolo_result.label
        mp_conf = mp_result.confidence
        yolo_conf = yolo_result.confidence
        
        # Vote based on confidence
        if mp_conf > yolo_conf:
            return DetectorOutput(
                label=mp_label,
                confidence=mp_conf,
                metadata={
                    'strategy': 'majority_mp_wins',
                    'mp_signal': mp_label,
                    'yolo_context': yolo_label,
                }
            )
        else:
            return DetectorOutput(
                label=yolo_label,
                confidence=yolo_conf,
                metadata={
                    'strategy': 'majority_yolo_wins',
                    'mp_signal': mp_label,
                    'yolo_context': yolo_label,
                }
            )
    
    def close(self):
        """Cleanup both detectors"""
        if self.mediapipe_detector:
            try:
                self.mediapipe_detector.close()
            except Exception as e:
                logger.warning(f"Error closing MediaPipe detector: {e}")
        
        if self.yolo_detector:
            try:
                self.yolo_detector.close()
            except Exception as e:
                logger.warning(f"Error closing YOLO detector: {e}")

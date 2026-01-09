"""MediaPipe-based detector wrapping motion and pose analysis"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging
import sys
import os

# Import from scripts/mediapipe_vlm if available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/mediapipe_vlm'))

from motion_detector import MediaPipeMotionDetector
from base import DetectorBase, DetectorOutput

logger = logging.getLogger(__name__)


class MediaPipeDetector(DetectorBase):
    """Activity detection via MediaPipe hand and pose tracking"""
    
    def __init__(self, confidence_threshold: float = 0.5, enable_visualization: bool = False):
        super().__init__(name='mediapipe', confidence_threshold=confidence_threshold)
        self.motion_detector = MediaPipeMotionDetector()
        self.enable_visualization = enable_visualization
        self.frame_count = 0
    
    def process_frame(self, frame) -> DetectorOutput:
        """
        Analyze frame for work activity using MediaPipe.
        
        Heuristic:
        - High hand motion + hands in work area = work
        - Active pose + hand velocity > threshold = work
        - Otherwise = idle
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        
        try:
            # Get hand detection
            hand_results = self.motion_detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_results = self.motion_detector.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Compute features
            hand_motion_regions, hand_velocity = self.motion_detector.detect_hand_motion(frame, hand_results)
            pose_activity = self.motion_detector.detect_pose_activity(pose_results)
            
            # Simple decision logic
            is_working = False
            decision_confidence = 0.0
            
            # Check hand motion (primary signal)
            if hand_velocity > 30:  # pixels per frame
                is_working = True
                decision_confidence = min(1.0, hand_velocity / 100.0)
            
            # Check pose (supporting signal)
            if pose_activity == 'active' and hand_motion_regions > 0:
                is_working = True
                decision_confidence = max(decision_confidence, 0.7)
            elif pose_activity == 'idle':
                is_working = False
                decision_confidence = max(decision_confidence, 0.5)
            
            label = 'work' if is_working and decision_confidence > self.confidence_threshold else 'idle'
            
            # Extract landmark coordinates for visualization
            hand_landmarks_list = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append({
                            'x': float(lm.x),
                            'y': float(lm.y),
                            'z': float(lm.z)
                        })
                    hand_landmarks_list.append(landmarks)
            
            pose_landmarks_list = []
            if pose_results.pose_landmarks:
                landmarks = []
                for lm in pose_results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z),
                        'visibility': float(lm.visibility) if hasattr(lm, 'visibility') else 1.0
                    })
                pose_landmarks_list = landmarks
            
            metadata = {
                'hand_velocity': float(hand_velocity),
                'hand_regions': int(hand_motion_regions),
                'pose_activity': str(pose_activity),
                'hands_detected': hand_results.multi_hand_landmarks is not None,
                'pose_detected': pose_results.pose_landmarks is not None,
                'confidence_threshold': self.confidence_threshold,
                # Store keypoints for visualization
                'hand_landmarks': hand_landmarks_list,
                'pose_landmarks': pose_landmarks_list,
            }
            
            return DetectorOutput(
                label=label,
                confidence=decision_confidence,
                metadata=metadata,
                raw_output={'hand_results': hand_results, 'pose_results': pose_results}
            )
        
        except Exception as e:
            logger.warning(f"MediaPipe detection error: {e}")
            return DetectorOutput(
                label='unknown',
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def close(self):
        """Clean up MediaPipe resources"""
        try:
            if self.motion_detector:
                if hasattr(self.motion_detector, 'hands') and self.motion_detector.hands:
                    self.motion_detector.hands.close()
                if hasattr(self.motion_detector, 'pose') and self.motion_detector.pose:
                    self.motion_detector.pose.close()
        except Exception as e:
            logger.warning(f"Error closing MediaPipe: {e}")

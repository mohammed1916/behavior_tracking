"""Test MediaPipe + YOLO detector integration"""

import sys
import os
import cv2
import numpy as np
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.detectors.mediapipe_yolo_detector import MediaPipeYoloDetector, MediaPipeYoloStrategy
from backend.detectors.yolo_detector import YOLODetector
from backend.detectors.mediapipe_detector import MediaPipeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_frame(width=640, height=480, frame_type='empty'):
    """Create a test frame for detector testing"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (255, 255, 255)  # White background
    
    if frame_type == 'hand_motion':
        # Add some motion (moving rectangle)
        cv2.circle(frame, (300, 200), 50, (0, 255, 0), -1)
        cv2.circle(frame, (350, 250), 50, (0, 255, 0), -1)
    elif frame_type == 'person_with_tools':
        # Simulate person with tools
        cv2.rectangle(frame, (100, 100), (200, 300), (100, 150, 200), -1)  # Person
        cv2.rectangle(frame, (210, 180), (250, 220), (200, 100, 100), -1)  # Tool
    elif frame_type == 'laptop':
        # Simulate laptop
        cv2.rectangle(frame, (100, 100), (400, 250), (100, 100, 100), -1)  # Laptop body
        cv2.rectangle(frame, (110, 110), (390, 240), (200, 200, 200), -1)  # Screen
    
    return frame


def test_mediapipe_detector():
    """Test MediaPipe detector in isolation"""
    logger.info("\n=== Testing MediaPipe Detector ===")
    try:
        detector = MediaPipeDetector(confidence_threshold=0.5)
        
        # Test with empty frame
        frame = create_test_frame(frame_type='empty')
        result = detector.process_frame(frame)
        logger.info(f"Empty frame: {result.label} (conf={result.confidence:.2f})")
        
        detector.close()
        logger.info(" MediaPipe detector works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe detector failed: {e}")
        return False


def test_yolo_detector():
    """Test YOLO detector in isolation"""
    logger.info("\n=== Testing YOLO Detector ===")
    try:
        detector = YOLODetector(
            model='yolov8n',
            confidence_threshold=0.5,
            device='cpu'  # Use CPU for testing
        )
        
        # Test with empty frame
        frame = create_test_frame(frame_type='empty')
        result = detector.process_frame(frame)
        logger.info(f"Empty frame: {result.label} (conf={result.confidence:.2f})")
        
        # Test with person frame
        frame = create_test_frame(frame_type='person_with_tools')
        result = detector.process_frame(frame)
        logger.info(f"Person frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Metadata: {result.metadata}")
        
        detector.close()
        logger.info(" YOLO detector works")
        return True
    except Exception as e:
        logger.error(f" YOLO detector failed: {e}")
        return False


def test_fusion_detector_cascade():
    """Test MediaPipe + YOLO detector with cascade strategy"""
    logger.info("\n=== Testing MediaPipe + YOLO Detector (CASCADE) ===")
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.CASCADE,
            device='cpu'
        )
        
        # Test with empty frame
        frame = create_test_frame(frame_type='empty')
        result = detector.process_frame(frame)
        logger.info(f"Empty frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Strategy: {result.metadata.get('strategy')}")
        
        # Test with person frame
        frame = create_test_frame(frame_type='person_with_tools')
        result = detector.process_frame(frame)
        logger.info(f"Person frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Strategy: {result.metadata.get('strategy')}")
        logger.info(f"  MP Signal: {result.metadata.get('mp_signal')}")
        logger.info(f"  YOLO Context: {result.metadata.get('yolo_context')}")
        
        detector.close()
        logger.info(" MediaPipe + YOLO detector (CASCADE) works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe + YOLO detector (CASCADE) failed: {e}")
        return False


def test_fusion_detector_weighted():
    """Test MediaPipe + YOLO detector with weighted strategy"""
    logger.info("\n=== Testing MediaPipe + YOLO Detector (WEIGHTED) ===")
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.WEIGHTED,
            device='cpu'
        )
        
        # Test with empty frame
        frame = create_test_frame(frame_type='empty')
        result = detector.process_frame(frame)
        logger.info(f"Empty frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Combined value: {result.metadata.get('combined_value'):.2f}")
        
        detector.close()
        logger.info(" MediaPipe + YOLO detector (WEIGHTED) works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe + YOLO detector (WEIGHTED) failed: {e}")
        return False


def test_detector_comparison():
    """Compare detector outputs side by side"""
    logger.info("\n=== Detector Comparison ===")
    
    frames = {
        'empty': create_test_frame(frame_type='empty'),
        'hand_motion': create_test_frame(frame_type='hand_motion'),
        'person_tools': create_test_frame(frame_type='person_with_tools'),
    }
    
    try:
        # Create detectors
        mp_detector = MediaPipeDetector(confidence_threshold=0.5)
        yolo_detector = YOLODetector(model='yolov8n', confidence_threshold=0.5, device='cpu')
        mediapipe_yolo_detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            strategy=MediaPipeYoloStrategy.CASCADE,
            device='cpu'
        )
        
        logger.info("\nFrame Type     | MediaPipe       | YOLO            | MediaPipe+YOLO")
        logger.info("-" * 70)
        
        for frame_name, frame in frames.items():
            mp_result = mp_detector.process_frame(frame)
            yolo_result = yolo_detector.process_frame(frame)
            combined_result = mediapipe_yolo_detector.process_frame(frame)
            
            mp_str = f"{mp_result.label}({mp_result.confidence:.1f})"
            yolo_str = f"{yolo_result.label}({yolo_result.confidence:.1f})"
            combined_str = f"{combined_result.label}({combined_result.confidence:.1f})"
            
            logger.info(f"{frame_name:14} | {mp_str:15} | {yolo_str:15} | {combined_str}")
        
        mp_detector.close()
        yolo_detector.close()
        mediapipe_yolo_detector.close()
        
        logger.info(" Detector comparison complete")
        return True
    except Exception as e:
        logger.error(f" Detector comparison failed: {e}")
        return False


if __name__ == '__main__':
    logger.info("Starting detector integration tests...\n")
    
    results = []
    
    # Test individual detectors
    results.append(("MediaPipe", test_mediapipe_detector()))
    results.append(("YOLO", test_yolo_detector()))
    
    # Test MediaPipe + YOLO strategies
    results.append(("MediaPipe+YOLO CASCADE", test_fusion_detector_cascade()))
    results.append(("MediaPipe+YOLO WEIGHTED", test_fusion_detector_weighted()))
    
    # Compare detectors
    results.append(("Detector Comparison", test_detector_comparison()))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        logger.info(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    logger.info("=" * 50)
    logger.info(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    sys.exit(0 if all_passed else 1)

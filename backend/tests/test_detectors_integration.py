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

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'assembly_drone')
TEST_VIDEOS = {
    # 'drone_assembly': os.path.join(TEST_DATA_DIR, 'drone_assembly.mp4'),
    'wire_screw': os.path.join(TEST_DATA_DIR, 'wire_screw.mp4'),
    # 'assembly_idle': os.path.join(TEST_DATA_DIR, 'assembly_idle.mp4'),
}


def load_test_frame_from_video(video_name: str, frame_index: int = 0):
    """Load a specific frame from a test video file"""
    video_path = TEST_VIDEOS.get(video_name)
    
    if not video_path or not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.warning(f"Failed to read frame {frame_index} from {video_name}")
        return None
    
    return frame


def test_mediapipe_detector():
    """Test MediaPipe detector in isolation with real video"""
    logger.info("\n=== Testing MediaPipe Detector ===")
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping MediaPipe test: no video available")
        return True
    
    try:
        detector = MediaPipeDetector(confidence_threshold=0.5)
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
        
        detector.close()
        logger.info(" MediaPipe detector works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe detector failed: {e}")
        return False


def test_yolo_detector():
    """Test YOLO detector in isolation with real video"""
    logger.info("\n=== Testing YOLO Detector ===")
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping YOLO test: no video available")
        return True
    
    try:
        detector = YOLODetector(
            model='yolov8n',
            confidence_threshold=0.5,
            device='cpu'
        )
        
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
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
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping CASCADE test: no video available")
        return True
    
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.CASCADE,
            device='cpu'
        )
        
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
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
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping WEIGHTED test: no video available")
        return True
    
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.WEIGHTED,
            device='cpu'
        )
        
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Combined value: {result.metadata.get('combined_value'):.2f}")
        
        detector.close()
        logger.info(" MediaPipe + YOLO detector (WEIGHTED) works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe + YOLO detector (WEIGHTED) failed: {e}")
        return False


def test_fusion_detector_consensus():
    """Test MediaPipe + YOLO detector with consensus strategy"""
    logger.info("\n=== Testing MediaPipe + YOLO Detector (CONSENSUS) ===")
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping CONSENSUS test: no video available")
        return True
    
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.CONSENSUS,
            device='cpu'
        )
        
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Strategy: {result.metadata.get('strategy')}")
        logger.info(f"  MP Signal: {result.metadata.get('mp_signal')}")
        logger.info(f"  YOLO Context: {result.metadata.get('yolo_context')}")
        
        detector.close()
        logger.info(" MediaPipe + YOLO detector (CONSENSUS) works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe + YOLO detector (CONSENSUS) failed: {e}")
        return False


def test_fusion_detector_majority_vote():
    """Test MediaPipe + YOLO detector with majority vote strategy"""
    logger.info("\n=== Testing MediaPipe + YOLO Detector (MAJORITY_VOTE) ===")
    
    frame = load_test_frame_from_video('wire_screw', frame_index=30)
    if frame is None:
        logger.warning(" Skipping MAJORITY_VOTE test: no video available")
        return True
    
    try:
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy=MediaPipeYoloStrategy.MAJORITY_VOTE,
            device='cpu'
        )
        
        result = detector.process_frame(frame)
        logger.info(f"Real frame: {result.label} (conf={result.confidence:.2f})")
        logger.info(f"  Strategy: {result.metadata.get('strategy')}")
        logger.info(f"  MP Signal: {result.metadata.get('mp_signal')}")
        logger.info(f"  YOLO Context: {result.metadata.get('yolo_context')}")
        
        detector.close()
        logger.info(" MediaPipe + YOLO detector (MAJORITY_VOTE) works")
        return True
    except Exception as e:
        logger.error(f" MediaPipe + YOLO detector (MAJORITY_VOTE) failed: {e}")
        return False


def test_real_video_frames():
    """Test detectors with real video frames"""
    logger.info("\n=== Testing with Real Video Files ===")
    
    test_videos = ['drone_assembly', 'wire_screw', 'assembly_idle']
    results = []
    
    for video_name in test_videos:
        # Try to load frame from video
        frame = load_test_frame_from_video(video_name, frame_index=30)  # Frame at 1 second (30fps)
        
        if frame is None:
            logger.warning(f"Skipping {video_name}: video not available")
            continue
        
        logger.info(f"\nTesting with {video_name}.mp4:")
        
        try:
            # Test with MediaPipeYolo detector
            detector = MediaPipeYoloDetector(
                yolo_model='yolov8n',
                strategy=MediaPipeYoloStrategy.CASCADE,
                device='cpu'
            )
            
            result = detector.process_frame(frame)
            logger.info(f"  Label: {result.label}")
            logger.info(f"  Confidence: {result.confidence:.2f}")
            logger.info(f"  Strategy: {result.metadata.get('strategy')}")
            logger.info(f"  MP Signal: {result.metadata.get('mp_signal')}")
            logger.info(f"  YOLO Context: {result.metadata.get('yolo_context')}")
            
            detector.close()
            results.append((video_name, True))
            
        except Exception as e:
            logger.error(f"  Failed to process {video_name}: {e}")
            results.append((video_name, False))
    
    if not results:
        logger.warning(" No real video files were tested (all skipped)")
        return True  # Don't fail test if videos aren't available
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        logger.info(" Real video frame tests passed")
    else:
        logger.warning(" Some real video tests failed")
    
    return all_passed


def test_detector_comparison():
    """Compare detector outputs side by side using real video frames"""
    logger.info("\n=== Detector Comparison ===")
    
    # Load frames from real video at different timestamps
    frame_indices = {'frame_30': 30, 'frame_60': 60, 'frame_90': 90}
    frames = {}
    
    for name, idx in frame_indices.items():
        frame = load_test_frame_from_video('wire_screw', frame_index=idx)
        if frame is not None:
            frames[name] = frame
    
    if not frames:
        logger.warning(" Skipping comparison: no video frames available")
        return True
    
    try:
        # Create detectors
        mp_detector = MediaPipeDetector(confidence_threshold=0.5)
        yolo_detector = YOLODetector(model='yolov8n', confidence_threshold=0.5, device='cpu')
        mediapipe_yolo_detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            strategy=MediaPipeYoloStrategy.CASCADE,
            device='cpu'
        )
        
        logger.info("\nFrame Index    | MediaPipe       | YOLO            | MediaPipe+YOLO")
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
    results.append(("MediaPipe+YOLO CONSENSUS", test_fusion_detector_consensus()))
    results.append(("MediaPipe+YOLO MAJORITY_VOTE", test_fusion_detector_majority_vote()))
    
    # Test with real video files
    results.append(("Real Video Frames", test_real_video_frames()))
    
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

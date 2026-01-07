"""
Test suite for detector modules.

Run with: pytest backend/tests/detectors/test_detectors.py -xvs
"""

import pytest
import numpy as np
import cv2
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDetectorBase:
    """Test the abstract base class"""
    
    def test_detector_output_creation(self):
        """DetectorOutput should store label, confidence, and metadata"""
        from backend.detectors import DetectorOutput
        
        output = DetectorOutput(
            label='work',
            confidence=0.95,
            metadata={'hand_velocity': 45.2, 'detector': 'test'}
        )
        
        assert output.label == 'work'
        assert output.confidence == 0.95
        assert output.metadata['hand_velocity'] == 45.2


class TestMediaPipeDetector:
    """Test MediaPipe detector"""
    
    @pytest.fixture
    def detector(self):
        """Create a MediaPipe detector for testing"""
        from backend.detectors import MediaPipeDetector
        detector = MediaPipeDetector(confidence_threshold=0.5)
        yield detector
        detector.close()
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """MediaPipe detector should initialize successfully"""
        assert detector is not None
        assert detector.name == 'mediapipe'
        assert detector.confidence_threshold == 0.5
    
    def test_process_frame_returns_detector_output(self, detector, dummy_frame):
        """process_frame should return DetectorOutput"""
        from backend.detectors import DetectorOutput
        
        output = detector.process_frame(dummy_frame)
        
        assert isinstance(output, DetectorOutput)
        assert output.label in ['work', 'idle', 'unknown']
        assert 0 <= output.confidence <= 1.0
        assert isinstance(output.metadata, dict)
    
    def test_detector_cleanup(self):
        """Detector should clean up resources"""
        from backend.detectors import MediaPipeDetector
        
        detector = MediaPipeDetector()
        detector.close()
        
        # Should not raise an error when closing again
        detector.close()


class TestYOLODetector:
    """Test YOLO detector"""
    
    @pytest.fixture
    def detector(self):
        """Create a YOLO detector for testing"""
        try:
            from backend.detectors import YOLODetector
            detector = YOLODetector(model='yolov8n', confidence_threshold=0.5)
            yield detector
            detector.close()
        except ImportError:
            pytest.skip("ultralytics not installed")
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """YOLO detector should initialize successfully"""
        assert detector is not None
        assert detector.name == 'yolo'
        assert detector.confidence_threshold == 0.5
    
    def test_process_frame_returns_detector_output(self, detector, dummy_frame):
        """process_frame should return DetectorOutput"""
        from backend.detectors import DetectorOutput
        
        output = detector.process_frame(dummy_frame)
        
        assert isinstance(output, DetectorOutput)
        assert output.label in ['work', 'idle', 'unknown']
        assert 0 <= output.confidence <= 1.0
        assert isinstance(output.metadata, dict)


class TestFusionEngine:
    """Test multi-detector fusion"""
    
    @pytest.fixture
    def dummy_detectors(self):
        """Create mock detectors for testing fusion"""
        from backend.detectors import DetectorBase, DetectorOutput
        
        class MockDetector1(DetectorBase):
            def process_frame(self, frame):
                return DetectorOutput('work', 0.8, {})
            def close(self):
                pass
        
        class MockDetector2(DetectorBase):
            def process_frame(self, frame):
                return DetectorOutput('work', 0.7, {})
            def close(self):
                pass
        
        return {
            'detector1': MockDetector1('detector1'),
            'detector2': MockDetector2('detector2'),
        }
    
    def test_fusion_engine_initialization(self, dummy_detectors):
        """Fusion engine should initialize with detectors"""
        from backend.detectors import FusionEngine, FusionMode
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.WEIGHTED)
        
        assert engine is not None
        assert len(engine.detectors) == 2
        assert engine.mode == FusionMode.WEIGHTED
    
    def test_consensus_fusion_all_agree(self, dummy_detectors):
        """Consensus mode should return work when all detectors agree on work"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.CONSENSUS)
        
        outputs = {
            'detector1': DetectorOutput('work', 0.8, {}),
            'detector2': DetectorOutput('work', 0.7, {}),
        }
        
        fused = engine.fuse(outputs)
        
        assert fused.label == 'work'
        assert fused.confidence > 0.7
    
    def test_consensus_fusion_disagree(self, dummy_detectors):
        """Consensus mode should return unknown when detectors disagree"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.CONSENSUS)
        
        outputs = {
            'detector1': DetectorOutput('work', 0.8, {}),
            'detector2': DetectorOutput('idle', 0.7, {}),
        }
        
        fused = engine.fuse(outputs)
        
        assert fused.label == 'unknown'
    
    def test_weighted_fusion(self, dummy_detectors):
        """Weighted mode should return fused label"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.WEIGHTED)
        
        outputs = {
            'detector1': DetectorOutput('work', 0.8, {}),
            'detector2': DetectorOutput('idle', 0.3, {}),
        }
        
        fused = engine.fuse(outputs)
        
        # Work score (0.8 + 0.0) vs Idle score (0 + 0.3)
        # Work should win
        assert fused.label == 'work'
        assert fused.confidence > 0
    
    def test_cascade_fusion_uses_primary(self, dummy_detectors):
        """Cascade mode should return primary detector output"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.CASCADE)
        
        outputs = {
            'mediapipe': DetectorOutput('work', 0.8, {}),
            'yolo': DetectorOutput('idle', 0.2, {}),
        }
        
        fused = engine.fuse(outputs)
        
        # MediaPipe is primary in cascade, so should return work
        # (because 0.8 > 0.6 threshold)
        assert fused.label == 'work'
    
    def test_majority_fusion(self, dummy_detectors):
        """Majority mode should return majority vote"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        engine = FusionEngine(dummy_detectors, mode=FusionMode.MAJORITY)
        
        outputs = {
            'detector1': DetectorOutput('work', 0.8, {}),
            'detector2': DetectorOutput('work', 0.7, {}),
        }
        
        fused = engine.fuse(outputs)
        
        assert fused.label == 'work'


class TestIntegration:
    """Integration tests with real detectors"""
    
    @pytest.fixture
    def test_frame(self):
        """Load or create a test frame"""
        # Create a simple test frame with some content
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some motion-like elements (bright spots)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        
        return frame
    
    def test_mediapipe_detector_on_test_frame(self, test_frame):
        """Test MediaPipe detector on a synthetic frame"""
        from backend.detectors import MediaPipeDetector
        
        detector = MediaPipeDetector()
        output = detector.process_frame(test_frame)
        detector.close()
        
        assert output is not None
        logger.info(f"MediaPipe output: {output.label}, confidence: {output.confidence}")
    
    def test_detector_sequential_processing(self, test_frame):
        """Test processing multiple frames sequentially"""
        from backend.detectors import MediaPipeDetector
        
        detector = MediaPipeDetector()
        
        outputs = []
        for i in range(3):
            output = detector.process_frame(test_frame)
            outputs.append(output)
        
        detector.close()
        
        assert len(outputs) == 3
        assert all(o is not None for o in outputs)


@pytest.mark.benchmark
class TestPerformance:
    """Benchmark detector performance"""
    
    @pytest.fixture
    def test_frame(self):
        """Create a realistic test frame (1080p)"""
        return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    
    def test_mediapipe_throughput(self, test_frame, benchmark):
        """Benchmark MediaPipe detector throughput"""
        from backend.detectors import MediaPipeDetector
        
        detector = MediaPipeDetector()
        
        def process():
            detector.process_frame(test_frame)
        
        result = benchmark(process)
        detector.close()
    
    def test_fusion_throughput(self, benchmark):
        """Benchmark fusion engine throughput"""
        from backend.detectors import FusionEngine, FusionMode, DetectorOutput
        
        outputs = {
            'detector1': DetectorOutput('work', 0.8, {}),
            'detector2': DetectorOutput('work', 0.7, {}),
        }
        
        engine = FusionEngine({}, mode=FusionMode.WEIGHTED)
        
        def fuse():
            engine.fuse(outputs)
        
        benchmark(fuse)


if __name__ == '__main__':
    pytest.main([__file__, '-xvs'])

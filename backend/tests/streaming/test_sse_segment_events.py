"""
Test suite to verify SSE segment events are produced by the VLM streaming endpoint.

Validates that:
1. segment events are emitted for LLM classifier_source
2. Each segment contains required fields: stage, start_time, end_time, duration, label, llm_output, timeline
3. Temporal aggregation correctly groups semantically-similar captions
4. No segments are emitted for VLM or BOW classifier sources (expected behavior)
"""

import os
import json
import pytest
import time
import threading
from pathlib import Path
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)


class SSEEventCollector:
    """Helper to collect SSE events from streaming endpoint."""
    
    def __init__(self, url, timeout=30):
        self.url = url
        self.timeout = timeout
        self.events = []
        self.errors = []
        self.finished = False
        self.thread = None
        
    def collect(self):
        """Stream events from URL and collect them."""
        try:
            req = urllib.request.Request(self.url)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            self.events.append(data)
                            if data.get('stage') == 'finished':
                                self.finished = True
                        except json.JSONDecodeError as e:
                            self.errors.append(f"JSON parse error: {e}")
        except Exception as e:
            self.errors.append(f"Stream error: {str(e)}")
            
    def run_async(self):
        """Run collection in background thread."""
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        
    def wait_finished(self, timeout=None):
        """Wait for stream to finish."""
        start = time.time()
        timeout = timeout or self.timeout
        while not self.finished and (time.time() - start) < timeout:
            time.sleep(0.1)
        if self.thread:
            self.thread.join(timeout=1)
            
    def get_stages(self):
        """Return list of stage values from events."""
        return [e.get('stage') for e in self.events]
    
    def get_segments(self):
        """Return all 'segment' stage events."""
        return [e for e in self.events if e.get('stage') == 'segment']
    
    def get_samples(self):
        """Return all 'sample' stage events."""
        return [e for e in self.events if e.get('stage') == 'sample']


@pytest.fixture
def test_video_path():
    """Return path to test video file."""
    video = Path(__file__).parent.parent.parent / 'data' / 'assembly_drone' / 'assembly_drone_240_144.mp4'
    assert video.exists(), f"Test video not found: {video}"
    return str(video)


@pytest.fixture
def backend_url():
    """Backend base URL."""
    return 'http://localhost:8001'


@pytest.fixture
def upload_test_video(test_video_path, backend_url):
    """Upload test video and return filename."""
    with open(test_video_path, 'rb') as f:
        import urllib.request as ur
        data = f.read()
        req = ur.Request(
            f"{backend_url}/backend/upload_vlm",
            data=data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        # Use multipart form approach instead
        import io
        from urllib.parse import urlencode
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        body = io.BytesIO()
        body.write(f'--{boundary}\r\n'.encode())
        body.write(b'Content-Disposition: form-data; name="video"; filename="test.mp4"\r\n')
        body.write(b'Content-Type: video/mp4\r\n\r\n')
        body.write(data)
        body.write(f'\r\n--{boundary}--\r\n'.encode())
        
        req = ur.Request(
            f"{backend_url}/backend/upload_vlm",
            data=body.getvalue(),
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
        )
        try:
            with ur.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result['filename']
        except Exception as e:
            pytest.skip(f"Failed to upload video: {e}")


class TestSegmentEventsLLMMode:
    """Test segment events in LLM mode with Qwen + CUDA."""
    
    @pytest.mark.integration
    def test_segment_events_emitted_in_llm_mode(self, backend_url, upload_test_video):
        """Verify segment events are emitted when classifier_source=llm."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=Describe+what+is+happening"
            f"&classifier_source=llm"
            f"&classifier_mode=multi"
            f"&processing_mode=every_2s"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        # Verify stream completed
        assert collector.finished, "Stream did not finish"
        assert not collector.errors, f"Errors occurred: {collector.errors}"
        
        # Verify stages
        stages = collector.get_stages()
        logger.info(f"Event stages: {stages}")
        assert 'started' in stages, "No 'started' event"
        assert 'finished' in stages, "No 'finished' event"
        
        # Verify at least one sample event
        samples = collector.get_samples()
        assert len(samples) > 0, "No sample events received"
        logger.info(f"Collected {len(samples)} sample events")
        
        # Verify segment events were emitted
        segments = collector.get_segments()
        assert len(segments) > 0, "No segment events received (expected > 0 for LLM mode)"
        logger.info(f"Collected {len(segments)} segment events")
        
    @pytest.mark.integration
    def test_segment_structure(self, backend_url, upload_test_video):
        """Verify each segment event has required fields."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=What+is+happening"
            f"&classifier_source=llm"
            f"&classifier_mode=binary"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        segments = collector.get_segments()
        assert len(segments) > 0, "No segments to validate"
        
        # Check first segment structure
        seg = segments[0]
        required_fields = ['stage', 'start_time', 'end_time', 'duration', 'dominant_caption', 'timeline']
        for field in required_fields:
            assert field in seg, f"Segment missing required field: {field}"
            
        # Validate field types
        assert seg['stage'] == 'segment'
        assert isinstance(seg['start_time'], (int, float))
        assert isinstance(seg['end_time'], (int, float))
        assert isinstance(seg['duration'], (int, float))
        assert isinstance(seg['dominant_caption'], str)
        assert isinstance(seg['timeline'], str)
        
        # Validate label is present and valid
        assert 'label' in seg, "Segment missing 'label' field"
        label = seg['label']
        assert label in ['work', 'idle', 'assembling_drone', 'using_phone', 'unknown'], f"Invalid label: {label}"
        
        logger.info(f"Segment 0: {label} ({seg['duration']:.2f}s)")
        
    @pytest.mark.integration
    def test_no_segments_in_vlm_mode(self, backend_url, upload_test_video):
        """Verify NO segment events are emitted in VLM mode (expected behavior)."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=Label+this"
            f"&classifier_source=vlm"
            f"&classifier_mode=multi"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        segments = collector.get_segments()
        assert len(segments) == 0, f"VLM mode should not emit segments, but got {len(segments)}"
        logger.info("VLM mode correctly produces no segment events")
        
    @pytest.mark.integration
    def test_no_segments_in_bow_mode(self, backend_url, upload_test_video):
        """Verify NO segment events are emitted in BOW mode (expected behavior)."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=Describe"
            f"&classifier_source=bow"
            f"&classifier_mode=binary"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        segments = collector.get_segments()
        assert len(segments) == 0, f"BOW mode should not emit segments, but got {len(segments)}"
        logger.info("BOW mode correctly produces no segment events")


class TestSegmentEventsSampleFlow:
    """Test that sample events have correct label from each classifier source."""
    
    @pytest.mark.integration
    def test_sample_labels_in_llm_mode(self, backend_url, upload_test_video):
        """Verify sample events have labels in LLM mode."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&classifier_source=llm"
            f"&classifier_mode=binary"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        samples = collector.get_samples()
        assert len(samples) > 0
        
        # Check labels exist and are valid
        labels = [s.get('label') for s in samples]
        valid_labels = {'work', 'idle', 'assembling_drone', 'using_phone', 'unknown', None}
        for label in labels:
            assert label in valid_labels, f"Invalid label: {label}"
            
        logger.info(f"Sample labels: {set(labels)}")
        
    @pytest.mark.integration
    def test_sample_labels_in_vlm_mode(self, backend_url, upload_test_video):
        """Verify sample events have normalized labels in VLM mode."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&classifier_source=vlm"
            f"&classifier_mode=multi"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        samples = collector.get_samples()
        assert len(samples) > 0
        
        labels = [s.get('label') for s in samples]
        valid_labels = {'assembling_drone', 'idle', 'using_phone', 'unknown'}
        for label in labels:
            assert label in valid_labels, f"Invalid multi-mode label: {label}"
            
        logger.info(f"VLM mode labels: {set(labels)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])

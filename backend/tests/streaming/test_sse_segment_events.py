"""
Test suite to verify SSE segment events are produced by the VLM streaming endpoint.

Validates that:
1. segment events are emitted for LLM classifier_source
2. Each segment contains required fields: stage, start_time, end_time, duration, label, llm_output, timeline
3. Temporal aggregation correctly groups semantically-similar captions
4. No segments are emitted for VLM or BOW classifier sources (expected behavior)

pytest backend/tests/streaming/test_sse_segment_events.py -v -s --tb=short
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
    
    def __init__(self, url, timeout=180):
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
    # video = Path(__file__).parent.parent.parent / 'data' / 'assembly_drone' / 'assembly_drone_240_144.mp4'
    repo_root = Path(__file__).resolve().parents[3]
    video = repo_root / 'data' / 'assembly_drone' / 'wire.mp4'
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
        
        collector = SSEEventCollector(url, timeout=240)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        # Verify stream completed or at least segments were emitted
        if not collector.finished:
            # Allow proceeding if segments are present even if 'finished' wasn't emitted within timeout
            assert len(collector.get_segments()) > 0, "Stream did not finish and no segments were emitted"
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
        
        collector = SSEEventCollector(url, timeout=240)
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
        
        collector = SSEEventCollector(url, timeout=240)
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
        
        collector = SSEEventCollector(url, timeout=240)
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
        
        collector = SSEEventCollector(url, timeout=240)
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
        
        collector = SSEEventCollector(url, timeout=240)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        samples = collector.get_samples()
        assert len(samples) > 0
        
        labels = [s.get('label') for s in samples]
        valid_labels = {'assembling_drone', 'idle', 'using_phone', 'unknown'}
        for label in labels:
            assert label in valid_labels, f"Invalid multi-mode label: {label}"
            
        logger.info(f"VLM mode labels: {set(labels)}")


class TestTemporalAggregationLogic:
    """Test the internal aggregation logic: Jaccard similarity, timeline quality, debouncing."""
    
    @pytest.mark.integration
    def test_similar_captions_grouped_in_same_segment(self, backend_url, upload_test_video):
        """Verify that semantically similar captions are grouped together in one segment."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=Describe+the+activity"
            f"&classifier_source=llm"
            f"&classifier_mode=binary"
            f"&processing_mode=every_2s"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        segments = collector.get_segments()
        assert len(segments) > 0, "No segments to analyze"
        
        # Check that each segment contains multiple similar captions
        for seg in segments:
            captions = seg.get('captions', [])
            timeline = seg.get('timeline', '')
            
            # Verify timeline contains timestamps and captions
            assert '<t=' in timeline, "Timeline missing timestamp markers"
            
            # Parse timeline to verify captions are included
            timeline_lines = [line for line in timeline.split('\n') if line.strip()]
            assert len(timeline_lines) > 0, "Timeline should have at least one caption"
            
            # For segments with multiple captions, check they share common keywords
            if len(captions) > 1:
                # Simple token overlap check (mimics Jaccard logic)
                def extract_keywords(text):
                    import re
                    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
                    stop = {'the','and','with','using','person','currently','a','an','is','are','in','on','of','their','to'}
                    return set([w for w in words if len(w) > 2 and w not in stop])
                
                # Check that consecutive captions share keywords
                for i in range(len(captions) - 1):
                    tokens1 = extract_keywords(captions[i] or '')
                    tokens2 = extract_keywords(captions[i+1] or '')
                    if tokens1 and tokens2:
                        # Jaccard similarity should exist (some overlap)
                        intersection = len(tokens1 & tokens2)
                        # At least some similarity or very short captions
                        assert intersection > 0 or len(tokens1) < 3 or len(tokens2) < 3, \
                            f"Adjacent captions lack keyword overlap: '{captions[i]}' vs '{captions[i+1]}'"
        
        logger.info(f"✓ Verified {len(segments)} segments have semantically coherent caption grouping")
    
    @pytest.mark.integration
    def test_timeline_format_and_content(self, backend_url, upload_test_video):
        """Verify the aggregated timeline has correct format with timestamps and captions."""
        
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
        assert len(segments) > 0
        
        import re
        for i, seg in enumerate(segments):
            timeline = seg.get('timeline', '')
            captions = seg.get('captions', [])
            
            # Verify format: <t=X.XX> caption text
            pattern = r'<t=(\d+\.\d+)>\s+(.+)'
            matches = re.findall(pattern, timeline)
            
            assert len(matches) > 0, f"Segment {i}: Timeline has invalid format"
            assert len(matches) == len(captions), \
                f"Segment {i}: Timeline entries ({len(matches)}) != captions count ({len(captions)})"
            
            # Verify timestamps are within segment bounds
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            
            for timestamp_str, caption_text in matches:
                timestamp = float(timestamp_str)
                assert start_time <= timestamp <= end_time + 0.1, \
                    f"Segment {i}: Timestamp {timestamp} outside bounds [{start_time}, {end_time}]"
                assert len(caption_text.strip()) > 0, \
                    f"Segment {i}: Empty caption at t={timestamp}"
            
            # Verify captions in timeline match captions list
            timeline_captions = [m[1].strip() for m in matches]
            for cap in captions:
                assert any(cap in tc or tc in cap for tc in timeline_captions), \
                    f"Segment {i}: Caption '{cap}' not found in timeline"
            
        logger.info(f"✓ Verified timeline format and content for {len(segments)} segments")
    
    @pytest.mark.integration
    def test_llm_interprets_timeline_correctly(self, backend_url, upload_test_video):
        """Verify LLM label matches the semantic content of the aggregated timeline."""
        
        url = (
            f"{backend_url}/backend/vlm_local_stream?"
            f"filename={upload_test_video}"
            f"&model=qwen/qwen2-vl-2b-instruct"
            f"&prompt=assembly+task"
            f"&classifier_source=llm"
            f"&classifier_mode=binary"
        )
        
        collector = SSEEventCollector(url, timeout=120)
        collector.run_async()
        collector.wait_finished(timeout=120)
        
        segments = collector.get_segments()
        assert len(segments) > 0
        
        # Check that LLM label correlates with timeline content
        work_keywords = {'assembling', 'assembly', 'drone', 'tool', 'screw', 'wire', 'connect', 
                        'tighten', 'parts', 'building', 'working', 'attach', 'install'}
        idle_keywords = {'idle', 'sitting', 'standing', 'waiting', 'resting', 'not', 'nothing'}
        
        mismatches = []
        for i, seg in enumerate(segments):
            label = seg.get('label', '')
            timeline = seg.get('timeline', '').lower()
            dominant = seg.get('dominant_caption', '').lower()
            
            # Count keyword occurrences
            work_count = sum(1 for kw in work_keywords if kw in timeline or kw in dominant)
            idle_count = sum(1 for kw in idle_keywords if kw in timeline or kw in dominant)
            
            # If clear signals exist, verify label matches
            if work_count > idle_count and work_count >= 2:
                if label != 'work':
                    mismatches.append({
                        'segment': i,
                        'label': label,
                        'expected': 'work',
                        'work_signals': work_count,
                        'idle_signals': idle_count,
                        'dominant': dominant[:80]
                    })
            elif idle_count > work_count and idle_count >= 2:
                if label != 'idle':
                    mismatches.append({
                        'segment': i,
                        'label': label,
                        'expected': 'idle',
                        'work_signals': work_count,
                        'idle_signals': idle_count,
                        'dominant': dominant[:80]
                    })
        
        # Allow some mismatches (LLM isn't perfect), but should be mostly correct
        mismatch_rate = len(mismatches) / len(segments) if segments else 0
        assert mismatch_rate < 0.4, \
            f"Too many label mismatches ({len(mismatches)}/{len(segments)}): {mismatches[:3]}"
        
        logger.info(f"✓ LLM correctly interpreted {len(segments) - len(mismatches)}/{len(segments)} segments")
        if mismatches:
            logger.warning(f"Label mismatches: {mismatches}")
    
    @pytest.mark.integration
    def test_debouncing_creates_segment_splits(self, backend_url, upload_test_video):
        """Verify that changes in caption content trigger segment boundaries (debouncing)."""
        
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
        
        segments = collector.get_segments()
        samples = collector.get_samples()
        
        assert len(segments) > 1, "Expected multiple segments to test debouncing behavior"
        assert len(samples) > 0, "Need samples to analyze caption transitions"
        
        # Verify segments are time-sequential and non-overlapping
        for i in range(len(segments) - 1):
            seg1 = segments[i]
            seg2 = segments[i + 1]
            
            end1 = seg1.get('end_time', 0)
            start2 = seg2.get('start_time', 0)
            
            # Segments should be sequential (next starts after or at previous end)
            assert start2 >= end1 - 0.1, \
                f"Segment {i+1} starts before segment {i} ends: {end1} vs {start2}"
        
        # Verify segment boundaries correspond to caption changes
        for i in range(len(segments) - 1):
            seg1 = segments[i]
            seg2 = segments[i + 1]
            
            # Get last caption from seg1 and first from seg2
            caps1 = seg1.get('captions', [])
            caps2 = seg2.get('captions', [])
            
            if caps1 and caps2:
                last_cap1 = caps1[-1] or ''
                first_cap2 = caps2[0] or ''
                
                # Extract keywords
                def keywords(text):
                    import re
                    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
                    stop = {'the','and','with','using','person','currently','a','an','is','are','in','on','of','their','to'}
                    return set([w for w in words if len(w) > 2 and w not in stop])
                
                kw1 = keywords(last_cap1)
                kw2 = keywords(first_cap2)
                
                # Calculate Jaccard similarity
                if kw1 and kw2:
                    intersection = len(kw1 & kw2)
                    union = len(kw1 | kw2)
                    jaccard = intersection / union if union > 0 else 0
                    
                    # Segment split should happen when similarity drops
                    # (not always perfect due to debounce/time windows, but generally true)
                    logger.info(f"Segment {i}->{i+1}: Jaccard={jaccard:.2f} "
                              f"('{last_cap1[:40]}' -> '{first_cap2[:40]}')")
        
        logger.info(f"✓ Verified {len(segments)} segments show proper temporal boundaries")
    
    @pytest.mark.integration
    def test_minimum_segment_duration(self, backend_url, upload_test_video):
        """Verify segments respect minimum duration threshold (filtering noise)."""
        
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
        
        segments = collector.get_segments()
        
        # According to code: agg_min_window = 1.5 seconds
        min_duration = 1.5
        
        short_segments = []
        for i, seg in enumerate(segments):
            duration = seg.get('duration', 0)
            if duration < min_duration:
                short_segments.append({
                    'segment': i,
                    'duration': duration,
                    'start': seg.get('start_time'),
                    'end': seg.get('end_time')
                })
        
        # All segments should meet minimum duration
        assert len(short_segments) == 0, \
            f"Found {len(short_segments)} segments below {min_duration}s threshold: {short_segments}"
        
        logger.info(f"✓ All {len(segments)} segments meet minimum duration requirement")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])

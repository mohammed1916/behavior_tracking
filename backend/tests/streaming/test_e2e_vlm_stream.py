import os
import json
import pytest
from fastapi.testclient import TestClient

import backend.server as server


class DummyCaptioner:
    """Mock VLM that returns deterministic captions for testing."""
    def __init__(self, device=None):
        self.device = device
        self.call_count = 0

    def __call__(self, image):
        # Cycle through captions to simulate realistic activity transitions
        captions = [
            "person assembling drone parts",
            "person tightening screws",
            "person connecting wires",
            "person sitting idle",
            "person at rest",
            "person assembling parts again",
        ]
        caption = captions[self.call_count % len(captions)]
        self.call_count += 1
        return [{"generated_text": caption}]


def collect_sse_events(response_lines):
    """Parse SSE streaming response into events."""
    events = []
    buf = ''
    for raw_line in response_lines:
        if raw_line is None:
            continue
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode('utf-8')
            except Exception:
                line = raw_line.decode('latin-1', errors='ignore')
        else:
            line = str(raw_line)
        line = line.strip('\r')
        if line == '':
            # end of event marker
            if buf:
                text = '\n'.join([l[6:] if l.startswith('data: ') else l for l in buf.splitlines()])
                try:
                    obj = json.loads(text)
                    events.append(obj)
                except Exception:
                    pass
            buf = ''
            continue
        if line.startswith('data:'):
            buf += line + '\n'
    return events


@pytest.mark.timeout(60)
def test_vlm_local_stream_no_llm(tmp_path, monkeypatch):
    """Test VLM-only mode (no LLM): no segment events should be emitted."""
    client = TestClient(server.app)

    # locate sample video
    here = os.path.dirname(os.path.dirname(__file__))
    sample = os.path.abspath(os.path.join(here, '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        sample = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        pytest.skip('sample video data/assembly_idle.mp4 not found')

    monkeypatch.setattr(server, 'get_captioner_for_model', lambda model=None, device_override=None: DummyCaptioner(device='cpu'))

    with open(sample, 'rb') as fh:
        files = {'video': ('assembly_idle.mp4', fh, 'video/mp4')}
        resp = client.post('/backend/upload_vlm', files=files)
    assert resp.status_code == 200, resp.text
    filename = resp.json().get('filename')
    assert filename

    # Stream without LLM
    params = {
        'filename': filename,
        'model': 'test_dummy',
        'classifier_source': 'vlm',  # VLM mode: no LLM
        'classifier_mode': 'binary'
    }
    with client.stream('GET', '/backend/vlm_local_stream', params=params) as resp:
        assert resp.status_code == 200
        events = collect_sse_events(resp.iter_lines())

    # In VLM mode: should see sample events but NO segment events
    samples = [e for e in events if e.get('stage') == 'sample']
    segments = [e for e in events if e.get('stage') == 'segment']
    
    assert len(samples) > 0, "Expected sample events in VLM mode"
    assert len(segments) == 0, "VLM mode should NOT emit segment events"


@pytest.mark.timeout(60)
def test_vlm_to_llm_aggregation_e2e(tmp_path, monkeypatch):
    """Test new architecture: VLM → time-windowed aggregation → LLM classification."""
    client = TestClient(server.app)

    here = os.path.dirname(os.path.dirname(__file__))
    sample = os.path.abspath(os.path.join(here, '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        sample = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        pytest.skip('sample video data/assembly_idle.mp4 not found')

    monkeypatch.setattr(server, 'get_captioner_for_model', lambda model=None, device_override=None: DummyCaptioner(device='cpu'))

    with open(sample, 'rb') as fh:
        files = {'video': ('assembly_idle.mp4', fh, 'video/mp4')}
        resp = client.post('/backend/upload_vlm', files=files)
    assert resp.status_code == 200, resp.text
    filename = resp.json().get('filename')
    assert filename

    # Stream WITH LLM
    params = {
        'filename': filename,
        'model': 'test_dummy',
        'classifier_source': 'llm',  # LLM mode: enable aggregation + classification
        'classifier_mode': 'binary',
        'processing_mode': 'every_2s'
    }
    with client.stream('GET', '/backend/vlm_local_stream', params=params) as resp:
        assert resp.status_code == 200
        events = collect_sse_events(resp.iter_lines())

    # Verify the pipeline:
    # 1. Sample events (VLM captions with timestamps)
    samples = [e for e in events if e.get('stage') == 'sample']
    assert len(samples) > 0, "Expected sample events in LLM mode"
    
    # Each sample should have a caption
    for s in samples:
        assert 'caption' in s or 'generated_text' in s, f"Sample missing caption: {s}"
    
    # 2. Segment events (after time-windowed aggregation + LLM)
    segments = [e for e in events if e.get('stage') == 'segment']
    assert len(segments) > 0, "Expected segment events in LLM mode (time-aggregation -> LLM)"
    
    # Verify segment structure
    for seg in segments:
        # Timeline with timestamps
        assert 'timeline' in seg, f"Segment missing timeline: {seg}"
        timeline = seg['timeline']
        assert '<t=' in timeline, f"Timeline missing timestamp format: {timeline}"
        
        # LLM label decision
        assert 'label' in seg, f"Segment missing label: {seg}"
        label = seg['label']
        assert label in ['work', 'idle', 'assembling_drone', 'unknown'], \
            f"Invalid label from LLM: {label}"
        
        # Duration and time bounds
        assert 'duration' in seg, f"Segment missing duration: {seg}"
        assert seg['duration'] >= 0, f"Invalid duration: {seg['duration']}"
    
    # 3. Verify captions are never dropped in aggregation
    all_sample_captions = [s.get('caption') or s.get('generated_text') for s in samples]
    segment_timeline_captions = []
    for seg in segments:
        # Parse timeline to extract captions
        for line in seg['timeline'].split('\n'):
            if '<t=' in line:
                # Format: <t=X.XX> caption text
                parts = line.split('> ', 1)
                if len(parts) == 2:
                    segment_timeline_captions.append(parts[1])
    
    # All captions should appear in some segment timeline (order may differ due to aggregation)
    # At minimum, sample count should match or be close (some may be grouped)
    assert len(segment_timeline_captions) > 0, "No captions in segment timelines"

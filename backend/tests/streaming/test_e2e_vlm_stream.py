import os
import json
import pytest
from fastapi.testclient import TestClient

import backend.server as server


class DummyCaptioner:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, image):
        # Return a structure compatible with `_normalize_caption_output`
        return [{"generated_text": "dummy caption for testing"}]


@pytest.mark.timeout(30)
def test_vlm_local_stream_e2e(tmp_path, monkeypatch):
    client = TestClient(server.app)

    # locate sample video
    here = os.path.dirname(os.path.dirname(__file__))
    sample = os.path.abspath(os.path.join(here, '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        # try alternative path relative to repo root
        sample = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'assembly_idle.mp4'))
    if not os.path.exists(sample):
        pytest.skip('sample video data/assembly_idle.mp4 not found')

    # Monkeypatch the server captioner loader to return a lightweight dummy
    monkeypatch.setattr(server, 'get_captioner_for_model', lambda model=None, device_override=None: DummyCaptioner(device='cpu'))

    # Upload video via the upload endpoint
    with open(sample, 'rb') as fh:
        files = {'video': ('assembly_idle.mp4', fh, 'video/mp4')}
        resp = client.post('/backend/upload_vlm', files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    filename = data.get('filename')
    assert filename, 'upload did not return filename'

    # Connect to vlm_local_stream with streaming enabled
    params = {
        'filename': filename,
        'model': 'test_dummy',
        'prompt': '',
        'use_llm': 'false'
    }
    url = '/backend/vlm_local_stream'
    got_video_info = False
    got_finished = False
    buf = ''
    with client.stream('GET', url, params=params) as resp:
        assert resp.status_code == 200
        for raw_line in resp.iter_lines():
            if raw_line is None:
                continue
            # resp.iter_lines returns bytes
            if isinstance(raw_line, bytes):
                try:
                    line = raw_line.decode('utf-8')
                except Exception:
                    line = raw_line.decode('latin-1', errors='ignore')
            else:
                line = str(raw_line)
            line = line.strip('\r')
            if line == '':
                # end of event
                if buf:
                    # strip leading 'data: ' if present
                    text = '\n'.join([l[6:] if l.startswith('data: ') else l for l in buf.splitlines()])
                    try:
                        obj = json.loads(text)
                    except Exception:
                        buf = ''
                        continue
                    stage = obj.get('stage')
                    if stage == 'video_info':
                        got_video_info = True
                    if stage == 'finished':
                        got_finished = True
                        break
                buf = ''
                continue
            # accumulate only lines starting with data: (ignore event:)
            if line.startswith('data:'):
                buf += line + '\n'
    assert got_video_info, 'did not receive video_info SSE event'
    assert got_finished, 'did not receive finished SSE event'

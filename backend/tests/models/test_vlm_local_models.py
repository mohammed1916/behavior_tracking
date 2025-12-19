import pytest
from fastapi.testclient import TestClient

import backend.server as server
import backend.captioner as captioner


def test_vlm_local_models_available_toggle():
    client = TestClient(server.app)

    # Ensure we start with an empty captioner cache (in-place clear so server reference stays valid)
    captioner._captioner_cache.clear()

    resp = client.get("/backend/vlm_local_models")
    assert resp.status_code == 200
    models = resp.json().get("models", [])
    assert isinstance(models, list)

    if not models:
        pytest.skip("no models configured in local_vlm_models.json")

    mid = models[0]["id"]

    # Initially should be unavailable
    m = next((x for x in models if x.get('id') == mid), None)
    assert m is not None
    assert m.get('available') in (False, None) or m.get('available') is False

    # Insert a dummy captioner into the cache for the exact id
    class DummyCaptioner:
        def __init__(self, device=None):
            self.device = device

    captioner._captioner_cache[mid] = DummyCaptioner(device='cpu')

    resp2 = client.get("/backend/vlm_local_models")
    assert resp2.status_code == 200
    models2 = resp2.json().get('models', [])
    m2 = next((x for x in models2 if x.get('id') == mid), None)
    assert m2 is not None
    assert m2.get('available') is True
    assert m2.get('device') == 'cpu'

    # Clear and test lowercase key matches
    captioner._captioner_cache.clear()
    captioner._captioner_cache[mid.lower()] = DummyCaptioner(device='cuda:0')

    resp3 = client.get('/backend/vlm_local_models')
    m3 = next((x for x in resp3.json().get('models', []) if x.get('id') == mid), None)
    assert m3 is not None
    assert m3.get('available') is True
    assert m3.get('device') == 'cuda:0'

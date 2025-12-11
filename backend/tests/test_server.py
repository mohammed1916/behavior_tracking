import os
import io
import base64
import importlib.util
from PIL import Image
from fastapi.testclient import TestClient


# Load the server module by path so tests work regardless of PYTHONPATH
here = os.path.dirname(__file__)
server_path = os.path.join(here, "..", "server.py")
spec = importlib.util.spec_from_file_location("server_mod", server_path)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

client = TestClient(server.app)


class DummyCaptioner:
    def __call__(self, image):
        return [{"generated_text": "dummy caption"}]


def make_test_image_b64():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_vlm_local_models_list():
    resp = client.get("/backend/vlm_local_models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "models" in data


def test_load_vlm_model_with_mock(monkeypatch):
    # Patch the loader to return a dummy captioner and avoid heavy model loads
    monkeypatch.setattr(server, "get_captioner_for_model", lambda model_id: DummyCaptioner())
    resp = client.post("/backend/load_vlm_model", data={"model": "test_model"})
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("loaded") is True or j.get("status") == "ok"


def test_caption_endpoint_with_mock(monkeypatch):
    monkeypatch.setattr(server, "get_captioner_for_model", lambda model_id: DummyCaptioner())
    b64 = make_test_image_b64()
    payload = {"image": f"data:image/jpeg;base64,{b64}", "model": "test_model"}
    resp = client.post("/backend/caption", json=payload)
    assert resp.status_code == 200
    j = resp.json()
    assert "caption" in j
    assert j["caption"] == "dummy caption"


def test_caption_endpoint_model_missing(monkeypatch):
    # Simulate missing model loader
    monkeypatch.setattr(server, "get_captioner_for_model", lambda model_id: None)
    b64 = make_test_image_b64()
    payload = {"image": f"data:image/jpeg;base64,{b64}", "model": "nonexistent"}
    resp = client.post("/backend/caption", json=payload)
    assert resp.status_code == 500

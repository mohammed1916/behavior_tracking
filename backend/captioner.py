import os
import json
import logging
from typing import Any, Optional, Dict

import torch

from backend.vlm_blip import get_blip_captioner
from backend.vlm_qwen import QwenVLMAdapter


# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

_captioner_cache: Dict[str, Any] = {}

LOCAL_VLM_MODELS: Dict[str, dict] = {}
LOCAL_VLM_PROVIDERS: set[str] = set()


# -----------------------------------------------------------------------------
# Load JSON model registry (single source of truth)
# -----------------------------------------------------------------------------

_MODEL_CFG_PATH = os.path.join(os.path.dirname(__file__), "local_vlm_models.json")

with open(_MODEL_CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

for entry in cfg.get("models", []):
    model_id = (entry.get("id") or "").strip().lower()
    if not model_id:
        continue

    LOCAL_VLM_MODELS[model_id] = entry
    LOCAL_VLM_PROVIDERS.add(model_id.split("/")[0])

logging.info(
    "Loaded local VLM models from %s: %s",
    _MODEL_CFG_PATH,
    list(LOCAL_VLM_MODELS.keys()),
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_device(device_override: Optional[str]) -> str:
    """
    Resolve compute device deterministically.
    """
    if device_override:
        s = device_override.lower()
        if s.startswith("cuda") or "gpu" in s:
            return device_override
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"


def _cache_under_keys(obj: Any, *keys: str) -> None:
    """
    Cache an object under multiple normalized keys.
    """
    for k in keys:
        if k:
            _captioner_cache[k] = obj
            _captioner_cache[k.lower()] = obj


# -----------------------------------------------------------------------------
# Captioner loaders
# -----------------------------------------------------------------------------

def get_local_captioner() -> Optional[Any]:
    """
    Return the local BLIP captioner pipeline (cached, deterministic).
    """
    cached = _captioner_cache.get("blip")
    if cached is not None:
        return cached

    try:
        captioner = get_blip_captioner()
    except Exception:
        logging.exception("Failed to initialize BLIP captioner")
        return None

    _cache_under_keys(
        captioner,
        "blip",
        "Salesforce/blip-image-captioning-large",
    )

    logging.info(
        "Loaded local BLIP captioner (Salesforce/blip-image-captioning-large)"
    )
    return captioner


def get_captioner_for_model(
    model_id: Optional[str],
    device_override: Optional[str] = None,
) -> Optional[Any]:
    """
    Return a captioner for the requested model using exact JSON lookup.

    - No provider fallbacks
    - No silent retries
    - Cache-first resolution
    """
    if not model_id:
        return get_local_captioner()

    # Fast path: cache hit
    cached = _captioner_cache.get(model_id) or _captioner_cache.get(model_id.lower())
    if cached is not None:
        return cached

    mid = model_id.lower()
    cfg = LOCAL_VLM_MODELS.get(mid)
    if cfg is None:
        return None

    provider = mid.split("/")[0]

    # ------------------------------------------------------------------
    # Qwen
    # ------------------------------------------------------------------
    if provider == "qwen":
        device = _resolve_device(device_override)

        # Allow model registry to specify a preferred mode (e.g. "label")
        preferred_model_id = cfg.get("id") or model_id
        preferred_mode = cfg.get("mode", "caption")

        adapter = QwenVLMAdapter(
            model_id=preferred_model_id,
            device=device,
            mode=preferred_mode,
        )

        # Cache under both requested key and the registry id
        _cache_under_keys(adapter, model_id, cfg.get("id"))

        logging.info(
            "Loaded QwenVLMAdapter for %s (device=%s mode=%s)",
            preferred_model_id,
            device,
            preferred_mode,
        )
        return adapter

    # ------------------------------------------------------------------
    # BLIP / Salesforce
    # ------------------------------------------------------------------
    if provider == "salesforce" or "blip" in mid:
        return get_local_captioner()

    return None

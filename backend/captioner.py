import os
import json
import logging
from typing import Any, Optional

import torch
from transformers import pipeline

from backend.vlm_qwen import QwenVLMAdapter


_captioner_cache: dict = {}


# Load supported local VLM models from JSON (single source of truth)
LOCAL_VLM_MODELS: dict = {}
LOCAL_VLM_PROVIDERS: set = set()
_model_cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
with open(_model_cfg_path, 'r', encoding='utf-8') as _f:
    _cfg = json.load(_f)
for _e in _cfg.get('models', []):
    _mid = (_e.get('id') or '').lower()
    if not _mid:
        continue
    LOCAL_VLM_MODELS[_mid] = _e
    _prov = _mid.split('/')[0]
    LOCAL_VLM_PROVIDERS.add(_prov)
logging.info('Loaded local VLM models from %s: %s', _model_cfg_path, list(LOCAL_VLM_MODELS.keys()))


def get_local_captioner() -> Optional[Any]:
    """Return the local BLIP captioner pipeline (deterministic).

    This function assumes the `transformers` package is available and will
    raise ImportError otherwise.
    """
    global _captioner_cache
    if 'blip' in _captioner_cache:
        return _captioner_cache['blip']

    device = 0 if torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available() else -1
    p = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large', device=device)
    # Cache under a few common keys so callers probing different ids match
    _captioner_cache['blip'] = p
    try:
        # canonical transformers id used when creating the pipeline
        canonical = 'Salesforce/blip-image-captioning-large'
        _captioner_cache[canonical] = p
        _captioner_cache[canonical.lower()] = p
    except Exception:
        pass
    logging.info('Loaded local BLIP captioner (Salesforce/blip-image-captioning-large)')
    return p


def get_captioner_for_model(model_id: Optional[str], device_override: Optional[str] = None) -> Optional[Any]:
    """Return a captioner for the requested model using exact JSON lookup.

    No fallbacks or try/except are used: failures will raise exceptions.
    """
    if not model_id:
        return get_local_captioner()

    mid = model_id.lower()

    # Exact JSON-configured model id match first
    if mid in LOCAL_VLM_MODELS:
        cfg = LOCAL_VLM_MODELS[mid]
        prov = mid.split('/')[0]
        if prov == 'qwen':
            device = 'cpu'
            if device_override:
                s = str(device_override).lower()
                if s.startswith('cuda') or 'gpu' in s:
                    device = device_override
            else:
                if torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available():
                    device = 'cuda:0'
            # model_path = os.path.join(os.path.dirname(__file__), 'scripts', 'qwen_vlm_2b_activity_model')
            # adapter = QwenVLMAdapter(model_path, device=device)
            adapter = QwenVLMAdapter("Qwen/Qwen2-VL-2B-Instruct", device=device)
            # Cache adapter under both original and lowercased keys to help probes
            _captioner_cache[model_id] = adapter
            try:
                _captioner_cache[model_id.lower()] = adapter
            except Exception:
                pass
            logging.info('Loaded QwenVLMAdapter for %s (device=%s) via JSON config', model_id, device)
            return adapter
        if prov == 'salesforce' or 'blip' in mid:
            return get_local_captioner()

    # No provider-level fallback; only exact JSON-configured models are supported
    return None


def _normalize_caption_output(captioner, out):
    first = out[0] if isinstance(out, list) and len(out) > 0 else out
    for k in ('generated_text', 'caption', 'text'):
        if hasattr(first, 'get') and first.get(k):
            return first.get(k)
    ids_list = None
    if hasattr(first, '__contains__') and 'input_ids' in first:
        ids = first.get('input_ids') if hasattr(first, 'get') else None
        if ids is not None:
            if hasattr(ids, 'tolist'):
                ids_list = ids.tolist()
            else:
                ids_list = list(ids)
        if ids_list and hasattr(captioner, 'tokenizer'):
            return captioner.tokenizer.decode(ids_list, skip_special_tokens=True)
    return str(first)

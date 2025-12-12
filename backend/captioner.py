import os
import re
import json
import logging
import importlib
from typing import Optional
try:
    import torch
    from transformers import pipeline
except Exception:
    torch = None
    pipeline = None

QwenVLMAdapter = None
try:
    # try package-relative
    from .vlm_qwen import QwenVLMAdapter  # type: ignore
except Exception:
    try:
        from vlm_qwen import QwenVLMAdapter  # type: ignore
    except Exception:
        # fallback to loading by file path if present
        try:
            vlm_path = os.path.join(os.path.dirname(__file__), 'vlm_qwen.py')
            if os.path.exists(vlm_path):
                spec = importlib.util.spec_from_file_location('vlm_qwen', vlm_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                QwenVLMAdapter = getattr(mod, 'QwenVLMAdapter', None)
        except Exception:
            QwenVLMAdapter = None

# Cache captioners by model id
_captioner_cache = {}
_captioner_status = {}

def get_local_captioner():
    """Return a default local captioner (BLIP) if available."""
    global _captioner_cache
    if _captioner_cache.get('default') is not None:
        return _captioner_cache.get('default')
    device = 0 if torch is not None and torch.cuda.is_available() else -1
    try:
        if pipeline is None:
            return None
        p = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
        _captioner_cache['default'] = p
        logging.info('Loaded default local captioner Salesforce/blip-image-captioning-large')
        return p
    except Exception:
        _captioner_cache['default'] = None
        return None

def get_captioner_for_model(model_id: str, device_override: Optional[str] = None):
    """Return a captioner pipeline for `model_id`."""
    global _captioner_cache, _captioner_status
    if not model_id:
        return get_local_captioner()

    if model_id in _captioner_cache:
        if ('openflamingo' in model_id.lower() or 'flamingo' in model_id.lower()) and _captioner_cache[model_id] is None:
            del _captioner_cache[model_id]
        else:
            return _captioner_cache[model_id]

    # device selection helper
    def _parse_device_override(o: Optional[str]):
        if not o:
            return None
        s = str(o).lower()
        if s in ('cpu', 'none', '-1'):
            return -1
        if s.startswith('cuda') or 'gpu' in s:
            import re as _re
            m = _re.match(r'cuda:(\d+)', s)
            if m:
                return int(m.group(1))
            return 0
        return None

    parsed_override = _parse_device_override(device_override)
    device_reason = None
    explicit_gpu_request = False
    if parsed_override is not None:
        device = parsed_override
        device_reason = f'override requested ({device_override})'
        if device >= 0:
            explicit_gpu_request = True
    else:
        if torch is not None and torch.cuda.is_available():
            device = 0
            device_reason = 'gpu_available'
        else:
            device = -1
            device_reason = 'no_cuda_visible'

    try:
        if device >= 0 and torch is not None:
            try:
                free, total = torch.cuda.mem_get_info(device)
            except Exception:
                free = None
                total = None
            MIN_GPU_FREE = 2 * 1024 * 1024 * 1024
            if free is not None and free < MIN_GPU_FREE:
                logging.info('GPU available but free memory low; preferring CPU')
                device = -1
    except Exception:
        device = -1

    # read local_vlm_models.json if available to determine task
    task = 'image-to-text'
    cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            for e in cfg.get('models', []):
                if e.get('id') == model_id:
                    task = e.get('task', task)
                    break
    except Exception:
        pass

    # Special-case Qwen local adapter
    mid = (model_id or '').lower()
    if mid == 'qwen_local' or mid.startswith('qwen'):
        if QwenVLMAdapter is not None:
            try:
                dev_arg = ('cpu' if device < 0 else f'cuda:{device}')
                try:
                    adapter = QwenVLMAdapter(os.path.join(os.path.dirname(__file__), 'scripts', 'qwen_vlm_2b_activity_model'), device=dev_arg)
                except TypeError:
                    adapter = QwenVLMAdapter(os.path.join(os.path.dirname(__file__), 'scripts', 'qwen_vlm_2b_activity_model'))
                _captioner_cache[model_id] = adapter
                _captioner_status[model_id] = {'device': dev_arg, 'reason': device_reason}
                logging.info('Loaded QwenVLMAdapter for model %s (device=%s)', model_id, dev_arg)
                return adapter
            except Exception as e:
                logging.exception('Failed to initialize QwenVLMAdapter for %s', model_id)
                if explicit_gpu_request:
                    raise RuntimeError(f'Requested GPU device but adapter init failed: {e}')
                return None
        logging.warning('QwenVLMAdapter not available; refusing to load non-HF model id %s', model_id)
        return None

    # Load HF pipeline
    try:
        if pipeline is None:
            raise RuntimeError('transformers.pipeline not available')
        p = pipeline(task, model=model_id, device=device)
        _captioner_cache[model_id] = p
        dev_str = ('cpu' if device < 0 else f'cuda:{device}')
        _captioner_status[model_id] = {'device': dev_str, 'reason': device_reason}
        logging.info('Loaded captioner for model %s (task=%s) on device=%s', model_id, task, dev_str)
        return p
    except Exception as e:
        if explicit_gpu_request:
            _captioner_status[model_id] = {'device': None, 'reason': f'explicit_gpu_request_failed: {e}'}
            logging.exception('Explicit GPU load requested but failed for model %s', model_id)
            raise RuntimeError(f'Requested GPU device but model load failed: {e}')
        try:
            p = pipeline(task, model=model_id, device=-1)
            _captioner_cache[model_id] = p
            _captioner_status[model_id] = {'device': 'cpu', 'reason': 'gpu_load_failed_fallback_cpu'}
            logging.info('GPU load failed; loaded captioner for model %s on CPU as fallback', model_id)
            return p
        except Exception:
            _captioner_status[model_id] = {'device': None, 'reason': 'failed_to_load'}
            raise


def _normalize_caption_output(captioner, out):
    try:
        first = None
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
        else:
            first = out
        try:
            for k in ('generated_text', 'caption', 'text'):
                if hasattr(first, 'get') and first.get(k):
                    return first.get(k)
            ids_list = None
            if hasattr(first, '__contains__') and 'input_ids' in first:
                ids = first.get('input_ids') if hasattr(first, 'get') else None
                if ids is not None:
                    try:
                        if hasattr(ids, 'tolist'):
                            ids_list = ids.tolist()
                        else:
                            ids_list = list(ids)
                    except Exception:
                        ids_list = None
                if ids_list and hasattr(captioner, 'tokenizer'):
                    try:
                        return captioner.tokenizer.decode(ids_list, skip_special_tokens=True)
                    except Exception:
                        pass
            return str(first)
        except Exception:
            return str(first)
    except Exception:
        logging.exception('Error normalizing caption output')
        return 'Error normalizing caption output'

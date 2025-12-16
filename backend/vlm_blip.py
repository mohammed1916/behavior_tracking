"""BLIP adapter module.

Provides a thin wrapper/factory for the local BLIP image->text pipeline so
the server can treat BLIP similarly to other adapter-style VLMs (e.g. Qwen).
"""
import logging
import torch
from transformers import pipeline


def get_blip_captioner(device_override: str = None):
    """Create and return a BLIP image-to-text pipeline.

    Returns the `transformers` pipeline object. Device selection follows the
    same heuristic as previous code: prefer GPU if available, otherwise CPU.
    """
    try:
        device = 0 if torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available() else -1
    except Exception:
        device = -1

    # Allow optional override (simple handling for strings like 'cpu' or 'cuda:0')
    if device_override:
        s = str(device_override).lower()
        if s == 'cpu':
            device = -1
        elif s.startswith('cuda') or 'gpu' in s:
            # keep using integer 0 for a single-GPU default when overriding
            try:
                # try to parse an index from 'cuda:0'
                if ':' in s:
                    device = int(s.split(':', 1)[1])
                else:
                    device = 0
            except Exception:
                device = 0

    logging.info('Initializing BLIP captioner (Salesforce/blip-image-captioning-large) on device=%s', device)
    p = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large', device=device)

    # Wrap the raw pipeline in a small adapter that accepts an optional
    # `prompt` keyword (frontend may supply extra context). We try a few
    # different call styles to remain compatible with transformers versions.
    def captioner(image, prompt: str = None, **kwargs):
        if prompt:
            # Try pipeline variants that may accept prompt/text parameters.
            try:
                return p(image, prompt=prompt, **kwargs)
            except TypeError:
                try:
                    return p(images=image, text=prompt, **kwargs)
                except TypeError:
                    try:
                        return p(image, text=prompt, **kwargs)
                    except Exception:
                        # Last-resort: ignore prompt and call normally
                        return p(image, **kwargs)
        else:
            return p(image, **kwargs)

    # Provide access to underlying pipeline attributes if needed
    try:
        captioner.tokenizer = getattr(p, 'tokenizer', None)
        captioner.device = getattr(p, 'device', None)
    except Exception:
        pass

    return captioner

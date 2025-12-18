"""
BLIP adapter module.

Provides a thin wrapper/factory for the local BLIP image->text pipeline so
the server can treat BLIP similarly to other adapter-style VLMs (e.g. Qwen).
"""

import logging
from typing import Optional

import torch
from transformers import pipeline

# from .llm import VLM_BASE_PROMPT_TEMPLATE  # intentionally unused for now


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_device(device_override: Optional[str]) -> int:
    """
    Resolve transformers pipeline device index.

    Returns:
        -1  -> CPU
         0+ -> CUDA device index
    """
    # Explicit override always wins
    if device_override:
        s = str(device_override).lower()
        if s == "cpu":
            return -1
        if s.startswith("cuda") or "gpu" in s:
            try:
                if ":" in s:
                    return int(s.split(":", 1)[1])
            except Exception:
                pass
            return 0

    # Auto-detect
    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass

    return -1


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_blip_captioner(device_override: Optional[str] = None):
    """
    Create and return a BLIP image-to-text pipeline wrapper.

    Returns:
        Callable captioner(image, **kwargs) -> pipeline output
    """
    device = _resolve_device(device_override)

    logging.info(
        "Initializing BLIP captioner (Salesforce/blip-image-captioning-large) "
        "on device=%s",
        device,
    )

    pipe = pipeline(
        task="image-to-text",
        model="Salesforce/blip-image-captioning-large",
        device=device,
    )

    def captioner(image, **kwargs):
        """
        Thin wrapper over transformers pipeline.

        Accepts image input and forwards all keyword arguments.
        Prompt handling is intentionally disabled for BLIP.
        """
        return pipe(image, **kwargs)

    # Optional introspection hooks (kept commented for cleanliness)
    # captioner.pipeline = pipe
    # captioner.device = device
    # captioner.tokenizer = getattr(pipe, "tokenizer", None)

    return captioner

"""
Self-contained Qwen VLM adapter (independent of mediapipe_vlm).

Supports two modes:
- mode="caption": returns natural language captions
- mode="label": returns normalized activity labels

Always returns:
    [{'generated_text': ...}]
"""

import os
import logging
from typing import Optional

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2

from .llm import VLM_BASE_PROMPT_TEMPLATE


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_QWEN_MODEL = os.environ.get(
    "QWEN_MODEL_ID",
    "Qwen/Qwen2-VL-2B-Instruct",
)

SUPPORTED_MODES = {"caption", "label"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------

class QwenVLMAdapter:
    """
    Adapter around a Qwen-style Image->Text model.

    Usage:
        QwenVLMAdapter(mode="caption") → caption
        QwenVLMAdapter(mode="label")   → label
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        mode: str = "caption",
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{mode}', must be one of {SUPPORTED_MODES}")

        self.mode = mode
        self.model_id = model_id or DEFAULT_QWEN_MODEL
        self.device = _resolve_device(device)

        logging.info(
            "Initializing QwenVLMAdapter: model=%s device=%s mode=%s",
            self.model_id,
            self.device,
            self.mode,
        )

        # Processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        # Model
        if self.device.type == "cuda":
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_prompt(self, user_prompt: Optional[str]) -> str:
        """
        Prompt logic:
        - Caption mode → descriptive VLM prompt
        - Label mode   → same prompt (rules live in postprocess)
        """
        return user_prompt or VLM_BASE_PROMPT_TEMPLATE

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(
        self,
        image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 64,
    ):
        """
        Run inference.

        Returns:
            [{'generated_text': caption_or_label}]
        """
        try:
            image = _to_pil(image)
            prompt = self._build_prompt(prompt)

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            text = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

            # Mode-specific behavior
            if self.mode == "label":
                text = self._postprocess(text)

            return [{"generated_text": text}]

        except Exception as e:
            logging.exception("QwenVLMAdapter inference failed: %s", e)
            return [{"generated_text": ""}]

    # ------------------------------------------------------------------
    # Label post-processing (ONLY used in label mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess(text: str) -> str:
        txt = text.lower()

        if "phone" in txt:
            return "using_phone"
        if "assemble" in txt or "drone" in txt:
            return "assembling_drone"
        if "idle" in txt:
            return "idle"

        return "unknown"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self):
        try:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
        except Exception:
            pass

"""
Self-contained Qwen VLM adapter (independent of mediapipe_vlm).

Always returns raw natural language captions:
    [{'generated_text': ...}]

Label normalization is handled centrally in rules.py via normalize_label_text().
"""

import os
import logging
from typing import Optional, Callable

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
    Always returns raw natural language captions.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_id = model_id or DEFAULT_QWEN_MODEL
        self.device = _resolve_device(device)

        print(
            "Initializing QwenVLMAdapter: model=%s device=%s",
            self.model_id,
            self.device,
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

    def __call__(
        self,
        image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 80,
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        """
        Run inference and return raw caption.

        Returns:
            [{'generated_text': caption}]
        """
        try:
            image = _to_pil(image)
            prompt = prompt or VLM_BASE_PROMPT_TEMPLATE
            prompt_text = (prompt or "").strip()
            if "Answer:" not in prompt_text and "Assistant:" not in prompt_text:
                prompt_text = prompt_text + "\nAnswer:"

            inputs = self.processor(
                text=prompt_text,
                images=image,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            # Generation settings to encourage non-empty output
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 8,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.05,
            }
            try:
                tok = getattr(self.processor, "tokenizer", None)
                eos_id = getattr(tok, "eos_token_id", None)
                pad_id = getattr(tok, "pad_token_id", None) or eos_id
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id
                if pad_id is not None:
                    gen_kwargs["pad_token_id"] = pad_id
            except Exception:
                if on_debug:
                    try:
                        on_debug("Qwen: unable to set special token IDs for generation.")
                    except Exception:
                        pass
                logging.warning("QwenVLMAdapter: unable to set special token IDs for generation.")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            # print(f"[Qwen] raw output_ids={output_ids}")

            # in_shape = inputs.get('input_ids', None).shape if inputs.get('input_ids', None) is not None else None
            # print(f"[Qwen] prompt token shape={in_shape}, output shape={tuple(output_ids.shape)}")

            # Some Qwen variants return the prompt + generation; strip the prompt portion
            try:
                prompt_len = inputs.get('input_ids', None).shape[1] if inputs.get('input_ids', None) is not None else 0
            except Exception:
                prompt_len = 0

            if prompt_len > 0:
                generated_only = output_ids[:, prompt_len:]
            else:
                generated_only = output_ids

            # If model returned no extra tokens beyond the prompt, fall back to full decode
            if generated_only.numel() == 0:
                generated_only = output_ids

            text = self.processor.decode(
                generated_only[0],
                skip_special_tokens=True,
            ).strip()
            try:
                if on_debug:
                    on_debug(f"Qwen decoded: {text[:200]}")
            except Exception:
                pass

            # if text == "":
            #     try:
            #         fallback_prompt = "Describe the visible activities and actions of the main person in detail.\nAnswer:"
            #         fb_inputs = self.processor(
            #             text=fallback_prompt,
            #             images=image,
            #             return_tensors="pt",
            #         )
            #         fb_inputs = {k: v.to(self.device) for k, v in fb_inputs.items()}
            #         with torch.inference_mode():
            #             fb_ids = self.model.generate(**fb_inputs, **gen_kwargs)
            #         text = self.processor.decode(fb_ids[0], skip_special_tokens=True).strip()
            #         print(f"[Qwen] fallback decoded text='{text[:200]}'")
            #     except Exception as fb_e:
            #         logging.warning("Qwen fallback generation failed: %s", fb_e)

            return [{"generated_text": text}]

        except Exception as e:
            logging.exception("QwenVLMAdapter inference failed: %s", e)
            return [{"generated_text": ""}]

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

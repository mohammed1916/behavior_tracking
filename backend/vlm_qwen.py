"""Self-contained Qwen VLM adapter (independent of mediapipe_vlm).

Provides `QwenVLMAdapter` with a pipeline-like callable interface
returning `[{'generated_text': ...}]` so it can be used by `server.py`.
"""
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2
from .llm import VLM_BASE_PROMPT_TEMPLATE

DEFAULT_QWEN_MODEL = os.environ.get('QWEN_MODEL_ID', 'Qwen/Qwen2-VL-2B-Instruct')


class QwenVLMAdapter:
    """Adapter around a Qwen-style Image->Text model.

    Usage:
        adapter = QwenVLMAdapter(model_id='qwen_local' or HF id)
        adapter(pil_image) -> [{'generated_text': '...'}]
    """
    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or DEFAULT_QWEN_MODEL
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load processor and model. Prefer GPU (fp16) when available, but fall
        # back gracefully to CPU to avoid crashing systems without GPU.
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load processor for model {self.model_id}: {e}")

        try:
            if self.device == 'cuda':
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        device_map='cuda',
                        trust_remote_code=True
                    )
                except Exception:
                    # Last-resort: load without device_map then move to cuda
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    try:
                        self.model.to('cuda')
                    except Exception:
                        pass
            else:
                # CPU path
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id,
                    trust_remote_code=True
                )
                try:
                    self.model.to('cpu')
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")

    def _build_prompt(self):
        return VLM_BASE_PROMPT_TEMPLATE

    def __call__(self, image, max_new_tokens=20):
        """Run inference on a PIL.Image (or array).

        Returns pipeline-like output: `[{'generated_text': '<label>'}]`.
        """
        try:
            # Accept numpy arrays or PIL images
            if not isinstance(image, Image.Image):
                # assume OpenCV BGR numpy array
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            prompt = self._build_prompt()
            # Prepare inputs
            inputs = self.processor(text=prompt, images=image, return_tensors='pt')
            # Move inputs to model device if possible
            try:
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            except Exception:
                pass

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            # decode
            try:
                raw = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
            except Exception:
                # fallback: use tokenizer if present
                try:
                    raw = self.model.config.processor_class.decode(output_ids[0], skip_special_tokens=True)
                except Exception:
                    raw = str(output_ids[0])

            # Post-process into normalized label (simple heuristics)
            txt = raw.lower()
            if 'phone' in txt:
                label = 'using_phone'
            elif 'assemble' in txt or 'drone' in txt:
                label = 'assembling_drone'
            elif 'idle' in txt:
                label = 'idle'
            else:
                label = 'unknown'

            return [{'generated_text': label}]
        except Exception as e:
            # don't propagate heavy exceptions to server; return empty result
            return [{'generated_text': ''}]

    def release(self):
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            torch.cuda.empty_cache()
        except Exception:
            pass

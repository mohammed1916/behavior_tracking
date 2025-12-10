"""Vision Language Model classifier for activity recognition"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2
from config import VLM_MODEL_NAME, ACTIVITY_ASSEMBLING_DRONE, ACTIVITY_IDLE, ACTIVITY_USING_PHONE, ACTIVITY_UNKNOWN


class VLMActivityClassifier:
    """Qwen2-VL based activity classifier"""
    
    def __init__(self, device=None):
        """Initialize VLM model and processor"""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, trust_remote_code=True)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        ).to(self.device)
    
    def _build_prompt(self):
        """Build the classification prompt"""
        return """<|vision_start|><|image_pad|><|vision_end|>

        You are an expert activity recognition model.

        Look ONLY at the MAIN PERSON in the image. Ignore all other people or objects.

        Classify their CURRENT ACTION into exactly ONE label from the following:

        1. assembling_drone → The person is working with tools, touching a drone, handling drone parts, connecting wires, tightening screws, or performing assembly actions.
        2. idle → The person is standing or sitting without doing any task, arms resting, not interacting with objects.
        3. using_phone → The person is clearly holding or interacting with a phone.
        4. unknown → If the activity cannot be confidently identified.

        Rules:
        - Do NOT guess.
        - Only output exactly one label: assembling_drone, idle, using_phone, or unknown.
        - Do not add any extra text, explanations, or repeats.
        - End your answer with "<|endoftext|>"

        Answer:
        """
    
    def classify(self, frame):
        """Classify activity in a frame
        
        Args:
            frame: OpenCV frame (BGR)
        
        Returns:
            str: Activity label (assembling_drone, idle, using_phone, unknown)
        """
        # Convert frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        prompt = self._build_prompt()
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=20)
        
        raw_result = self.processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        print(f"VLM Output: {raw_result}")
        
        # Get last non-empty line
        lines = [line.strip() for line in raw_result.splitlines() if line.strip()]
        result = lines[-1] if lines else ACTIVITY_UNKNOWN
        
        return result
    
    def post_process_vlm_result(self, vlm_result):
        """Convert raw VLM output to standard activity label"""
        if "phone" in vlm_result:
            return ACTIVITY_USING_PHONE
        elif "assemble" in vlm_result or "drone" in vlm_result:
            return ACTIVITY_ASSEMBLING_DRONE
        elif "idle" in vlm_result:
            return ACTIVITY_IDLE
        else:
            return ACTIVITY_UNKNOWN
    
    def release(self):
        """Release model from memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()

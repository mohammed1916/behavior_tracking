"""
Run from root: python scripts/vlm_rules_offline.py 
"""
prompt = """<|vision_start|><|image_pad|><|vision_end|>
You are a vision-language model specialized in recognizing HANDS-ON ELECTRONICS and DRONE ASSEMBLY tasks.

Focus ONLY on the MAIN PERSON and their HAND actions.

Look specifically for:
- soldering wires
- connecting motors or ESCs
- assembling drone frames
- attaching propellers
- holding tools (soldering iron, screwdriver, pliers)
- placing electronic components on a workbench

Ignore:
- background people
- screens unless actively used



Rules:
- Prefer assembling_drone if hands interact with wires, tools, or drone parts


Answer:
<|endoftext|>
"""

# Choose EXACTLY ONE label:
# assembling_drone | idle | using_phone | unknown

# - Output ONLY the label

"""
Run from root:
python scripts/vlm_rules_offline.py
"""

import cv2
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time
from datetime import datetime
import numpy as np
import os

# ------------------------------------------------------
# MODE SWITCH
# ------------------------------------------------------
MODE = "describe"     # "classify" or "describe"

# ------------------------------------------------------
# Load Qwen2-VL 2B
# ------------------------------------------------------
model_name = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
).to(device)

model.eval()

# ------------------------------------------------------
# Motion Detection (used only in classify mode)
# ------------------------------------------------------
prev_frame_gray = None
motion_history = []
MOTION_HISTORY_SIZE = 10

def analyze_motion_pattern(frame_gray, prev_gray):
    diff = cv2.absdiff(frame_gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hand_regions = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
    return len(hand_regions) >= 2

# ------------------------------------------------------
# VLM INFERENCE
# ------------------------------------------------------
last_vlm_latency = 0.0

def vlm_infer(frame):
    global prev_frame_gray, motion_history, last_vlm_latency

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # -------- CLASSIFICATION PROMPT --------
    if MODE == "classify":
        prompt = """<|vision_start|><|image_pad|><|vision_end|>
kz        You are an expert model for recognizing DRONE ASSEMBLY activities.

        Focus on the MAIN PERSON's hands.

        Choose ONE label only:
        assembling_drone | idle | using_phone | unknown

        Rules:
        - Prefer assembling_drone when hands interact with tools, wires, or drone parts
        - Output ONLY the label

        Answer:
        <|endoftext|>
        """

    # -------- DESCRIPTION PROMPT --------
    else:
        prompt = """<|vision_start|><|image_pad|><|vision_end|>
        Describe what the MAIN PERSON is doing in ONE short factual sentence.

        Focus on:
        - hands
        - tools
        - drone or electronic components

        Do not speculate.
        Do not explain.
        No labels.

        Answer:
        <|endoftext|>
        """

    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="pt"
    ).to(device)

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=25)
    last_vlm_latency = time.time() - start

    decoded = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

    # -------- CLASSIFY MODE POST-PROCESS --------
    if MODE == "classify":
        decoded = decoded.lower()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame_gray is not None:
            productive = analyze_motion_pattern(frame_gray, prev_frame_gray)
        else:
            productive = False

        prev_frame_gray = frame_gray

        if "phone" in decoded:
            return "using_phone"
        if any(k in decoded for k in ["drone", "solder", "wire", "assemble"]):
            return "assembling_drone"
        if productive:
            return "assembling_drone"
        return decoded if decoded in ["idle", "unknown"] else "idle"

    # -------- DESCRIPTION MODE --------
    return decoded

# ------------------------------------------------------
# Video Input
# ------------------------------------------------------
VIDEO_PATH = r"data/assembly_drone/assembly_drone_240_144.mp4"

print("Video exists:", os.path.exists(VIDEO_PATH))
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Failed to open video file")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "data_collection.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    fps,
    (width, height),
)

# ------------------------------------------------------
# Processing Loop
# ------------------------------------------------------
prev_infer_time = time.time()
current_output = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    now = time.time()

    if now - prev_infer_time >= 1.0:
        current_output = vlm_infer(frame)
        prev_infer_time = now
        print(f"[VLM] {current_output} | {last_vlm_latency:.2f}s")

    font_scale = min(width, height) / 400
    thickness = 1

    label = "Description" if MODE == "describe" else "Activity"

    cv2.putText(
        frame,
        f"{label}: {current_output}",
        (5, int(20 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness
    )

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

"""
Run from root: python scripts/vlm_rules_offline.py 
"""
import cv2
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time
from datetime import datetime
import numpy as np

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
# Motion Detection
# ------------------------------------------------------
prev_frame_gray = None
motion_history = []
MOTION_HISTORY_SIZE = 10

def analyze_motion_pattern(frame_gray, prev_gray):
    h, w = frame_gray.shape
    diff = cv2.absdiff(frame_gray, prev_gray)

    overall_motion = np.sum(diff) / (h * w)

    work_area = diff[h // 2:, :]
    upper_area = diff[:h // 2, :]

    work_motion = np.sum(work_area) / work_area.size
    upper_motion = np.sum(upper_area) / upper_area.size

    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    hand_regions = [c for c in contours if 100 < cv2.contourArea(c) < 5000]

    return {
        "overall": overall_motion,
        "work_area": work_motion,
        "upper_area": upper_motion,
        "hand_score": len(hand_regions),
        "work_ratio": work_motion / (upper_motion + 0.1),
    }

def classify_motion_as_productive(stats, history):
    history.append(stats)
    if len(history) > MOTION_HISTORY_SIZE:
        history.pop(0)

    if len(history) < 3:
        return False

    avg_work = np.mean([m["work_area"] for m in history[-5:]])
    avg_hand = np.mean([m["hand_score"] for m in history[-5:]])
    avg_ratio = np.mean([m["work_ratio"] for m in history[-5:]])
    consistency = np.std([m["work_area"] for m in history[-5:]])

    return (
        2.0 < avg_work < 15.0 and
        avg_hand >= 2 and
        avg_ratio >= 1.2 and
        consistency < 5.0
    )

# ------------------------------------------------------
# VLM Classification
# ------------------------------------------------------
last_vlm_latency = 0.0

def classify_activity(frame):
    global prev_frame_gray, motion_history, last_vlm_latency

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

Choose EXACTLY ONE label:
assembling_drone | idle | using_phone | unknown

Rules:
- Prefer assembling_drone if hands interact with wires, tools, or drone parts
- Output ONLY the label

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
        output_ids = model.generate(**inputs, max_new_tokens=10)
    last_vlm_latency = time.time() - start

    decoded = processor.decode(output_ids[0], skip_special_tokens=True).lower()
    result = decoded.splitlines()[-1].strip() if decoded else "unknown"

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    is_productive = False
    if prev_frame_gray is not None:
        stats = analyze_motion_pattern(frame_gray, prev_frame_gray)
        is_productive = classify_motion_as_productive(stats, motion_history)

    prev_frame_gray = frame_gray

    if "phone" in result:
        return "using_phone"

    if any(k in result for k in ["assemble", "drone", "solder", "wire"]):
        return "assembling_drone"

    if is_productive:
        return "assembling_drone"

    return result if result in ["idle", "unknown"] else "idle"

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
log_data = []
CSV_FILE = "activity_log.csv"

def log_activity(activity, start, end):
    log_data.append([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        activity,
        round(end - start, 2)
    ])
    pd.DataFrame(
        log_data,
        columns=["timestamp", "activity", "duration_sec"]
    ).to_csv(CSV_FILE, index=False)

# ------------------------------------------------------
# Video Input (OFFLINE)
# ------------------------------------------------------
VIDEO_PATH = "data/assembly_drone/assembly_drone_240_144.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Failed to open video file")

input_fps = cap.get(cv2.CAP_PROP_FPS)
if input_fps == 0:
    input_fps = 25

print("Input FPS:", input_fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reset motion state
prev_frame_gray = None
motion_history.clear()

# SAFE codec
out = cv2.VideoWriter(
    "data_collection.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    input_fps,
    (width, height),
)

# ------------------------------------------------------
# Processing Loop
# ------------------------------------------------------
current_activity = "unknown"
activity_start_time = time.time()
prev_classify_time = time.time()

written_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break

    now = time.time()

    if now - prev_classify_time >= 1.0:
        new_activity = classify_activity(frame)

        if new_activity != current_activity:
            log_activity(current_activity, activity_start_time, now)
            current_activity = new_activity
            activity_start_time = now

        prev_classify_time = now
        print(f"[VLM] {new_activity} | Latency {last_vlm_latency:.3f}s")

    cv2.putText(frame, f"Activity: {current_activity}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    written_frames += 1

print("Frames written:", written_frames)

log_activity(current_activity, activity_start_time, time.time())

cap.release()
out.release()
cv2.destroyAllWindows()

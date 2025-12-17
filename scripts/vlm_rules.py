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

    num_regions = len([c for c in contours if cv2.contourArea(c) > 50])
    hand_regions = [c for c in contours if 100 < cv2.contourArea(c) < 5000]

    return {
        "overall": overall_motion,
        "work_area": work_motion,
        "upper_area": upper_motion,
        "num_regions": num_regions,
        "hand_score": len(hand_regions),
        "work_ratio": work_motion / (upper_motion + 0.1),
    }

def classify_motion_as_productive(stats, history):
    history.append(stats)
    if len(history) > MOTION_HISTORY_SIZE:
        history.pop(0)

    if len(history) < 3:
        return False, 0, 0

    avg_work = np.mean([m["work_area"] for m in history[-5:]])
    avg_hand = np.mean([m["hand_score"] for m in history[-5:]])
    avg_ratio = np.mean([m["work_ratio"] for m in history[-5:]])
    consistency = np.std([m["work_area"] for m in history[-5:]])

    is_productive = (
        2.0 < avg_work < 15.0 and
        avg_hand >= 2 and
        avg_ratio >= 1.2 and
        consistency < 5.0
    )

    return is_productive, avg_work, avg_hand

# ------------------------------------------------------
# VLM Classification with Timing
# ------------------------------------------------------
last_vlm_latency = 0.0

def classify_activity(frame):
    global prev_frame_gray, motion_history, last_vlm_latency

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    prompt = """<|vision_start|><|image_pad|><|vision_end|>

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

    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="pt"
    ).to(device)

    start_vlm = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=20)
    end_vlm = time.time()

    last_vlm_latency = end_vlm - start_vlm

    decoded = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    ).lower()

    result = decoded.splitlines()[-1].strip() if decoded else "unknown"

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_stats = None
    is_productive = False

    if prev_frame_gray is not None:
        motion_stats = analyze_motion_pattern(frame_gray, prev_frame_gray)
        is_productive, _, _ = classify_motion_as_productive(
            motion_stats, motion_history
        )

    prev_frame_gray = frame_gray

    if "phone" in result:
        return "using_phone"

    if "assemble" in result or "drone" in result:
        return "assembling_drone"

    if motion_stats:
        if is_productive:
            return "assembling_drone"
        if motion_stats["overall"] < 2.0:
            return "idle"
        if motion_stats["overall"] < 8.0:
            return "simply_sitting"

    return result if result in ["idle", "unknown"] else "idle"

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
log_data = []
CSV_FILE = "activity_log.csv"

def log_activity(activity, start, end):
    duration = round(end - start, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append([timestamp, activity, duration])

    pd.DataFrame(
        log_data,
        columns=["timestamp", "activity", "duration_sec"]
    ).to_csv(CSV_FILE, index=False)

# ------------------------------------------------------
# Video Stream + FPS
# ------------------------------------------------------
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "data_collection_.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height),
)

current_activity = "unknown"
activity_start_time = time.time()
prev_classify_time = time.time()

prev_fps_time = time.time()
fps = 0.0
frame_count = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()

    if now - prev_fps_time >= 1.0:
        fps = frame_count / (now - prev_fps_time)
        frame_count = 0
        prev_fps_time = now

    if now - prev_classify_time >= 1.0:
        new_activity = classify_activity(frame)

        if new_activity != current_activity:
            log_activity(current_activity, activity_start_time, now)
            current_activity = new_activity
            activity_start_time = now

        prev_classify_time = now

        print(
            f"[VLM] Activity: {new_activity} | "
            f"Latency: {last_vlm_latency:.3f}s"
        )

    elapsed = int(now - activity_start_time)

    cv2.putText(frame, f"Activity: {current_activity}", (30, 40),
                font, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Elapsed: {elapsed}s", (30, 80),
                font, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (30, 120),
                font, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"VLM Latency: {last_vlm_latency:.2f}s", (30, 160),
                font, 1, (0, 0, 255), 2)

    cv2.imshow("Drone Assembly Monitoring", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

log_activity(current_activity, activity_start_time, time.time())

cap.release()
out.release()
cv2.destroyAllWindows()

import cv2
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel
from PIL import Image
import time
from datetime import datetime
import numpy as np
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------------------------------
# Model / processor setup (correct for InternVL2)
# ------------------------------------------------------
model_name = "OpenGVLab/InternVL2-2B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Load as AutoModel which will auto-select the right class
try:
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()
except Exception as e:
    print(f"Failed to load with AutoModel: {e}")
    # Try alternate loading
    from transformers import InternVLChatModel
    model = InternVLChatModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    ).eval()

# ------------------------------------------------------
# Motion analysis functions
# ------------------------------------------------------
def analyze_motion_pattern(curr_frame, prev_frame):
    diff = cv2.absdiff(curr_frame, prev_frame)
    motion_score = np.mean(diff)

    h, w = diff.shape
    work_area = diff[int(h * 0.5):h, int(w * 0.2):int(w * 0.8)]
    work_score = np.mean(work_area)

    hand_area = diff[:int(h * 0.5), int(w * 0.2):int(w * 0.8)]
    hand_score = np.mean(hand_area)

    return {
        "overall": motion_score,
        "work_area": work_score,
        "hand_score": hand_score,
        "work_ratio": (work_score + 1e-5) / (motion_score + 1e-5)
    }

def classify_motion_as_productive(motion_stats, history):
    history.append(motion_stats["work_area"])
    if len(history) > 10:
        history.pop(0)

    avg_work = np.mean(history)
    productive = avg_work > 10 and motion_stats["hand_score"] > 5
    return productive, avg_work, motion_stats["hand_score"]

prev_frame_gray = None
motion_history = []
MOTION_HISTORY_SIZE = 10

# ------------------------------------------------------
# InternVL2-based activity classification
# ------------------------------------------------------
def classify_activity(frame):
    global prev_frame_gray, motion_history

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    prompt = """
You are an expert activity recognition model.
Look ONLY at the MAIN PERSON in the image.

Classify their CURRENT ACTION into exactly ONE label:
- assembling_drone
- idle
- using_phone
- unknown

Rules:
- No guessing.
- Output only ONE label.
- No extra text.
"""

    with torch.inference_mode():
        try:
            # Convert PIL image to tensor format
            img_array = np.array(img) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            if device == "cuda":
                img_tensor = img_tensor.half()
            
            # Tokenize prompt only
            inputs = processor(prompt, return_tensors='pt').to(device)
            
            # Manually call model with image and text
            outputs = model(input_ids=inputs.input_ids, pixel_values=img_tensor, max_new_tokens=50)
            raw_result = processor.decode(outputs, skip_special_tokens=True).strip().lower()
        except:
            # Fallback: try simpler approach silently
            try:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
                if device == "cuda":
                    img_tensor = img_tensor.half()
                inputs = processor(text=prompt, return_tensors='pt').to(device)
                outputs = model.generate(input_ids=inputs.input_ids, pixel_values=img_tensor, max_new_tokens=50)
                raw_result = processor.decode(outputs[0], skip_special_tokens=True).strip().lower()
            except:
                raw_result = "unknown"

        print("#########################", raw_result)

    # Clean output
    lines = [line.strip() for line in raw_result.splitlines() if line.strip()]
    result = lines[-1] if lines else "unknown"

    # Motion analysis
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_stats = None
    is_productive_motion = False

    if prev_frame_gray is not None:
        motion_stats = analyze_motion_pattern(frame_gray, prev_frame_gray)
        is_productive_motion, avg_work_motion, hand_score = classify_motion_as_productive(
            motion_stats, motion_history
        )
        print(f"Motion - Overall: {motion_stats['overall']:.2f}, Work Area: {motion_stats['work_area']:.2f}, "
              f"Hand Score: {motion_stats['hand_score']}, Productive: {is_productive_motion}")

    prev_frame_gray = frame_gray

    # High-level fusion logic
    if "phone" in result:
        return "using_phone"
    if "assemble" in result or "drone" in result:
        return "assembling_drone"
    if motion_stats:
        if is_productive_motion:
            return "assembling_drone"
        if motion_stats['overall'] < 2.0:
            return "idle"
        if 2.0 <= motion_stats['overall'] < 8.0:
            return "simply_sitting"
        if motion_stats['overall'] > 8.0 and motion_stats['work_ratio'] < 1.0:
            return "simply_sitting"
    if "idle" in result:
        return "idle"
    elif "unknown" in result:
        return "simply_sitting"
    return "idle"

# ------------------------------------------------------
# Activity tracking + CSV logging setup
# ------------------------------------------------------
current_activity = "unknown"
activity_start_time = time.time()
log_data = []
CSV_FILE = "activity_log.csv"

IDLE_LIMIT = 10
PHONE_LIMIT = 10
DRONE_LIMIT = 20

def log_activity(activity, start, end):
    duration = round(end - start, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append([timestamp, activity, duration])
    df = pd.DataFrame(log_data, columns=["timestamp", "activity", "duration_sec"])
    df.to_csv(CSV_FILE, index=False)

# ------------------------------------------------------
# Real-time video stream
# ------------------------------------------------------
cap = cv2.VideoCapture(0)

# Try a few frames to see if camera is actually working
test_frames = 0
for i in range(5):
    ret, frame = cap.read()
    if ret:
        test_frames += 1
    else:
        break

if test_frames == 0:
    print("Error: Could not access camera (no frames captured). Camera may not be available.")
    print("Please ensure a camera is connected and functioning properly.")
    cap.release()
    sys.exit(1)

# Reset camera
cap.release()
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("data_collection_.mp4", fourcc, 30.0, (width, height))

prev_classify_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
alert_message = ""

PHONE_RESET_FRAMES = 5
phone_missing_frames = 0

print("Starting activity monitoring...press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - prev_classify_time >= 1.0:
        new_activity = classify_activity(frame)
        prev_classify_time = current_time

        # PHONE RESET LOGIC
        if current_activity == "using_phone":
            if new_activity != "using_phone":
                phone_missing_frames += 1
            else:
                phone_missing_frames = 0

            if phone_missing_frames >= PHONE_RESET_FRAMES:
                log_activity(current_activity, activity_start_time, current_time)
                current_activity = new_activity
                activity_start_time = current_time
                phone_missing_frames = 0

        else:
            if new_activity != current_activity:
                log_activity(current_activity, activity_start_time, current_time)
                current_activity = new_activity
                activity_start_time = current_time
                phone_missing_frames = 0

    elapsed = current_time - activity_start_time
    alert_message = ""
    if current_activity == "idle" and elapsed > IDLE_LIMIT:
        alert_message = f"Idle for {int(elapsed)} sec"
    if current_activity == "simply_sitting" and elapsed > IDLE_LIMIT:
        alert_message = f"Simply sitting for {int(elapsed)} sec"
    if current_activity == "using_phone" and elapsed > PHONE_LIMIT:
        alert_message = f"Phone usage for {int(elapsed)} sec"
    if current_activity == "assembling_drone" and elapsed > DRONE_LIMIT:
        alert_message = f"Drone usage limit exceeded"

    cv2.putText(frame, f"Activity: {current_activity}", (30, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {int(elapsed)} sec", (30, 80), font, 1, (255, 255, 0), 2)
    if alert_message:
        cv2.putText(frame, alert_message, (30, 120), font, 1, (0, 0, 255), 3)

    cv2.imshow("Drone Assembly Monitoring", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_activity(current_activity, activity_start_time, time.time())
cap.release()
out.release()
cv2.destroyAllWindows()

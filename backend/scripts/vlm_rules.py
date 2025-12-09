import cv2
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import time
from datetime import datetime
import numpy as np
# ------------------------------------------------------
# Load Qwen2-VL 2B (CORRECT)
# ------------------------------------------------------
model_name = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
).to(device)

# ------------------------------------------------------
# Classification using VLM model
# ------------------------------------------------------
prev_frame_gray = None

def classify_activity(frame):
    global prev_frame_gray

    # Convert frame to PIL image
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

    inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=20)

    raw_result = processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()
    print("#########################", raw_result)

    # last non-empty line
    lines = [line.strip() for line in raw_result.splitlines() if line.strip()]
    result = lines[-1] if lines else "unknown"

    # ----------------------------
    # MOTION DETECTION
    # ----------------------------
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion = 0

    if prev_frame_gray is not None:
        diff = cv2.absdiff(frame_gray, prev_frame_gray)
        motion = np.sum(diff) / (frame_gray.shape[0] * frame_gray.shape[1])

    prev_frame_gray = frame_gray
    MOTION_THRESHOLD = 3   # tune this

    print("Motion:", motion)

    # ----------------------------
    # FINAL DECISION LOGIC
    # ----------------------------
    if "phone" in result:
        return "using_phone"

    if "assemble" in result or "drone" in result:
        return "assembling_drone"

    # If VLM is unsure:
    if "idle" in result and motion > MOTION_THRESHOLD:
        return "simply_sitting"
    elif "unknown" in result and motion > MOTION_THRESHOLD:
        return "working"
    else:
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
drone_limit=20

def log_activity(activity, start, end):
    duration = round(end - start, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_data.append([timestamp, activity, duration])

    df = pd.DataFrame(log_data, columns=["timestamp", "activity", "duration_sec"])
    df.to_csv(CSV_FILE, index=False)

# ------------------------------------------------------
# Real-time video stream
# ------------------------------------------------------
cap = cv2.VideoCapture("drone.mp4")



# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
out = cv2.VideoWriter("data_collection_20251105_141950.997_.mp4", fourcc, 30.0, (width, height))

prev_classify_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
alert_message = ""

PHONE_RESET_FRAMES = 5
phone_missing_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run VLM once per second (important)
    if current_time - prev_classify_time >= 1.0:
        new_activity = classify_activity(frame)
        prev_classify_time = current_time

        # ----------------------------------------------------
        # PHONE TIMER RESET LOGIC
        # ----------------------------------------------------
        if current_activity == "using_phone":
            if new_activity != "using_phone":
                phone_missing_frames += 1
            else:
                phone_missing_frames = 0

            # Reset only if phone is missing for enough frames
            if phone_missing_frames >= PHONE_RESET_FRAMES:
                log_activity(current_activity, activity_start_time, current_time)
                current_activity = new_activity
                activity_start_time = current_time
                phone_missing_frames = 0

        else:
            # Normal activity change
            if new_activity != current_activity:
                log_activity(current_activity, activity_start_time, current_time)
                current_activity = new_activity
                activity_start_time = current_time
                phone_missing_frames = 0


    elapsed = current_time - activity_start_time
    alert_message = ""

    if current_activity == "working" and elapsed > IDLE_LIMIT:
        alert_message = f"pls assemble drone"

    if current_activity == "simply_sitting" and elapsed > IDLE_LIMIT:
        alert_message = f" simply_sitting for {int(elapsed)} sec"

    if current_activity == "idle" and elapsed > IDLE_LIMIT:
        alert_message = f"Idle for {int(elapsed)} sec"

    if current_activity == "using_phone" and elapsed > PHONE_LIMIT:
        alert_message = f"Phone usage for {int(elapsed)} sec"

    if current_activity == "assembling_drone" and elapsed > drone_limit:
        alert_message = f"drone usage limit exceeded"

    cv2.putText(frame, f"Activity: {current_activity}", (30, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Time: {int(elapsed)} sec", (30, 80), font, 1, (255, 255, 0), 2)

    if alert_message:
        cv2.putText(frame, alert_message, (30, 120), font, 1, (0, 255, 0), 4)

    cv2.imshow("Drone Assembly Monitoring", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_activity(current_activity, activity_start_time, time.time())

cap.release()
out.release()
cv2.destroyAllWindows()

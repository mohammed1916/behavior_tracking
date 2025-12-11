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
print(device)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
).to(device)

# ------------------------------------------------------
# Enhanced Motion Detection for Productive Work
# ------------------------------------------------------
prev_frame_gray = None
motion_history = []
MOTION_HISTORY_SIZE = 10

def analyze_motion_pattern(frame_gray, prev_gray):
    """Enhanced motion analysis with region-based detection"""
    h, w = frame_gray.shape
    
    # Calculate overall motion
    diff = cv2.absdiff(frame_gray, prev_gray)
    overall_motion = np.sum(diff) / (h * w)
    
    # Region-based motion analysis (work area detection)
    # Bottom half = work area (hands, tools, desk)
    # Top half = upper body (less relevant for work detection)
    work_area = diff[h//2:, :]
    upper_area = diff[:h//2, :]
    
    work_motion = np.sum(work_area) / work_area.size
    upper_motion = np.sum(upper_area) / upper_area.size
    
    # Detect localized motion (small, precise movements = productive work)
    # Apply threshold to get significant motion pixels
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours of motion regions
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze motion characteristics
    num_motion_regions = len([c for c in contours if cv2.contourArea(c) > 50])
    
    # Hand-like motion detection (small to medium regions)
    hand_motion_regions = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
    hand_motion_score = len(hand_motion_regions)
    
    return {
        'overall': overall_motion,
        'work_area': work_motion,
        'upper_area': upper_motion,
        'num_regions': num_motion_regions,
        'hand_score': hand_motion_score,
        'work_ratio': work_motion / (upper_motion + 0.1)  # Avoid division by zero
    }

def classify_motion_as_productive(motion_stats, motion_history):
    """Determine if motion pattern indicates productive work"""
    
    # Thresholds (tune these based on your environment)
    WORK_MOTION_MIN = 2.0      # Minimum motion in work area
    WORK_MOTION_MAX = 15.0     # Maximum (too much = not focused)
    HAND_SCORE_MIN = 2         # Minimum hand-like motion regions
    WORK_RATIO_MIN = 1.2       # Work area should have more motion than upper body
    
    # Add current stats to history
    motion_history.append(motion_stats)
    if len(motion_history) > MOTION_HISTORY_SIZE:
        motion_history.pop(0)
    
    # Analyze recent motion pattern (smoothing)
    if len(motion_history) >= 3:
        avg_work_motion = np.mean([m['work_area'] for m in motion_history[-5:]])
        avg_hand_score = np.mean([m['hand_score'] for m in motion_history[-5:]])
        avg_work_ratio = np.mean([m['work_ratio'] for m in motion_history[-5:]])
        consistency = np.std([m['work_area'] for m in motion_history[-5:]])
        
        # Productive work indicators:
        # 1. Moderate, consistent motion in work area
        # 2. Multiple small motion regions (hands working)
        # 3. More motion in lower half than upper half
        # 4. Consistent pattern (not erratic)
        
        is_productive = (
            WORK_MOTION_MIN < avg_work_motion < WORK_MOTION_MAX and
            avg_hand_score >= HAND_SCORE_MIN and
            avg_work_ratio >= WORK_RATIO_MIN and
            consistency < 5.0  # Consistent motion pattern
        )
        
        return is_productive, avg_work_motion, avg_hand_score
    
    return False, 0, 0

def classify_activity(frame):
    global prev_frame_gray, motion_history

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
    # ENHANCED MOTION DETECTION
    # ----------------------------
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

    # ----------------------------
    # FINAL DECISION LOGIC (Enhanced)
    # ----------------------------
    if "phone" in result:
        return "using_phone"

    if "assemble" in result or "drone" in result:
        return "assembling_drone"
    
    # Enhanced decision making with motion analysis
    if motion_stats:
        # Strong productive motion pattern detected
        if is_productive_motion:
            if "idle" in result or "unknown" in result:
                return "assembling_drone"  # Override VLM with motion evidence
            return "assembling_drone"
        
        # Minimal motion detected
        if motion_stats['overall'] < 2.0:
            return "idle"
        
        # Medium motion but not productive pattern
        if 2.0 <= motion_stats['overall'] < 8.0:
            if "idle" in result:
                return "simply_sitting"  # Some movement but not working
            elif "unknown" in result:
                return "simply_sitting"
        
        # High overall motion but not in work area (body movement)
        if motion_stats['overall'] > 8.0 and motion_stats['work_ratio'] < 1.0:
            return "simply_sitting"  # Moving but not working
    
    # Fallback to simple logic
    if "idle" in result:
        return "idle"
    elif "unknown" in result:
        return "simply_sitting"
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
# cap = cv2.VideoCapture("drone.mp4")
cap = cv2.VideoCapture(0)



# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
out = cv2.VideoWriter("data_collection_.mp4", fourcc, 30.0, (width, height))

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

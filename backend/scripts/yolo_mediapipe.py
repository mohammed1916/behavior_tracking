import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import os
import time

# --- Time Limits (in seconds) ---
TIME_LIMIT = 20   # set as desired

# Cooldown frames before resetting activity (anti-cheat)
RESET_DELAY_FRAMES = 10

activity_reset_counters = {
    "phone": 0,
    "cup": 0,
    "book": 0,
    "laptop": 0
}

# detection miss tolerance (frames) before zeroing CONFIRM counter
DETECT_RESET_FRAMES = 50
detection_miss_counters = {
    "phone": 0,
    "cup": 0,
    "book": 0,
    "laptop": 0
}

# Start timers when activity begins
activity_timers = {
    "phone": None,
    "cup": None,
    "book": None,
    "laptop": None
}

# Whether the limit already exceeded (stop repeated saving)
activity_flags = {
    "phone": False,
    "cup": False,
    "book": False,
    "laptop": False
}

# Save folder
SAVE_FOLDER = "exceeded_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -----------------------------
# Load YOLOv8 object detection model
# -----------------------------
object_model = YOLO("yolov8l.pt")  # COCO trained (adjust to your model path)

# -----------------------------
# Webcam and motion tracking
# -----------------------------
cap = cv2.VideoCapture(1)
prev_keypoints = None
movement_history = []

# Thresholds
MOTION_THRESHOLD = 5
STATIC_FRAMES = 30  # used for movement_history window

# Acting detection params
MIN_BODY_MOTION = 3.0      # body motion threshold for "real" activity
MIN_HAND_MOTION = 5.0      # hand motion threshold for "real" activity
ACTING_STATIC_FRAMES = 20  # consecutive low-motion frames to mark acting

acting_counters = {
    "phone": 0,
    "cup": 0,
    "book": 0,
    "laptop": 0
}

# -----------------------------
# Helper functions
# -----------------------------
def extract_keypoints(results_pose, results_hands, frame_shape):
    keypoints = []

    # -------- POSE (33 keypoints) --------
    if results_pose.pose_landmarks:
        for lm in results_pose.pose_landmarks.landmark:
            x = int(lm.x * frame_shape[1])
            y = int(lm.y * frame_shape[0])
            keypoints.append([x, y])
    else:
        keypoints += [[0, 0]] * 33

    # -------- HANDS (42 keypoints = 21 per hand) --------
    hand_kps = []
    if results_hands.multi_hand_landmarks:
        for hand in results_hands.multi_hand_landmarks:
            for lm in hand.landmark:
                x = int(lm.x * frame_shape[1])
                y = int(lm.y * frame_shape[0])
                hand_kps.append([x, y])

    # pad hand keypoints to EXACTLY 42
    while len(hand_kps) < 42:
        hand_kps.append([0, 0])

    # combine pose+hands
    keypoints += hand_kps

    # -------- ENFORCE EXACT SIZE (75 keypoints) --------
    keypoints = keypoints[:75]     # truncate if more
    return np.array(keypoints)


def draw_pose_mask(mask, keypoints):
    """Draw body + hands skeleton as mask"""
    body_connections = mp_pose.POSE_CONNECTIONS
    for connection in body_connections:
        a, b = connection
        if keypoints[a][0] > 0 and keypoints[b][0] > 0:
            cv2.line(mask, tuple(keypoints[a]), tuple(keypoints[b]), (255,255,255), 2)

    hand_connections = mp_hands.HAND_CONNECTIONS
    # Left hand: indices 33:54 (21 points)
    left_hand = keypoints[33:54]
    for connection in hand_connections:
        a, b = connection
        if left_hand[a][0] > 0 and left_hand[b][0] > 0:
            cv2.line(mask, tuple(left_hand[a]), tuple(left_hand[b]), (255,255,255), 2)
    # Right hand: indices 54:75
    right_hand = keypoints[54:75]
    for connection in hand_connections:
        a, b = connection
        if right_hand[a][0] > 0 and right_hand[b][0] > 0:
            cv2.line(mask, tuple(right_hand[a]), tuple(right_hand[b]), (255,255,255), 2)
    return mask

def point_to_box(point, box):
    if point is None:
        return 9999
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    return np.hypot(point[0]-cx, point[1]-cy)

# Persistent detection counters
task_memory = {"phone": 0, "cup": 0, "book": 0, "laptop": 0}
CONFIRM_FRAMES = 3  # required consecutive detections

PHONE_DIST = 150
BOTTLE_DIST = 150
BOOK_DIST = 180
LAPTOP_DIST = 200

def get_hand_centroid(keypoints):
    """Return average HAND position (more stable than wrist)."""
    left_hand = keypoints[33:54]
    right_hand = keypoints[54:75]

    def centroid(hand):
        valid = [pt for pt in hand if pt[0] > 0]
        if not valid:
            return None
        arr = np.array(valid)
        return np.mean(arr, axis=0)

    return centroid(left_hand), centroid(right_hand)

def update_activity_timer(activity, is_active, frame):
    global activity_timers, activity_flags, activity_reset_counters

    if is_active:
        # Detected this frame → cancel reset countdown
        activity_reset_counters[activity] = 0

        # Start timer if not already running
        if activity_timers[activity] is None:
            activity_timers[activity] = time.time()

        elapsed = time.time() - activity_timers[activity]

        # Check if time exceeded
        if elapsed >= TIME_LIMIT:
            if not activity_flags[activity]:
                ts = int(time.time())
                # Save a copy of frame to avoid race
                if frame is not None:
                    cv2.imwrite(f"{SAVE_FOLDER}/{activity}_{ts}.jpg", frame)
                activity_flags[activity] = True

            return True  # limit exceeded

        return False  # still ongoing

    else:
        # Not detected → increase cooldown counter
        activity_reset_counters[activity] += 1

        # Only reset if disappeared for enough frames
        if activity_reset_counters[activity] >= RESET_DELAY_FRAMES:
            activity_timers[activity] = None
            activity_flags[activity] = False
            activity_reset_counters[activity] = 0

        return False

def is_acting(activity, avg_motion, avg_hand_motion):
    """Return True if user is likely pretending (holding object but minimal motion)."""
    global acting_counters
    low_body = avg_motion < MIN_BODY_MOTION
    low_hand = avg_hand_motion < MIN_HAND_MOTION

    if low_body and low_hand:
        acting_counters[activity] += 1
    else:
        acting_counters[activity] = 0

    return acting_counters[activity] >= ACTING_STATIC_FRAMES

def detect_task(objects, keypoints, avg_motion, avg_hand_motion, frame):
    """
    Detect tasks and incorporate:
    - object proximity checks
    - persistent detection (task_memory)
    - acting detection (using avg_motion & avg_hand_motion)
    - activity timers (update_activity_timer)
    """
    global task_memory, detection_miss_counters

    left_hand, right_hand = get_hand_centroid(keypoints)
    nose = keypoints[0] if keypoints[0][0] > 0 else None

    # Detected flags for this frame
    detected_phone = False
    detected_cup = False
    detected_book = False
    detected_laptop = False

    # -----------------------------
    # 1️⃣ Using Phone (PRIORITY)
    # -----------------------------
    if "cell phone" in objects:
        phone = objects["cell phone"]
        if (left_hand is not None and point_to_box(left_hand, phone) < PHONE_DIST) or \
           (right_hand is not None and point_to_box(right_hand, phone) < PHONE_DIST):
            detected_phone = True

    # -----------------------------
    # 2️⃣ Bottle / Cup
    # -----------------------------
    cup = objects.get("cup") if objects.get("cup") is not None else objects.get("bottle")

    drinking = False
    if cup is not None:
        if (left_hand is not None and point_to_box(left_hand, cup) < BOTTLE_DIST) or \
           (right_hand is not None and point_to_box(right_hand, cup) < BOTTLE_DIST):
            detected_cup = True

        if nose is not None and point_to_box(nose, cup) < 150:
            drinking = True  # don't return early — use as label

    # -----------------------------
    # 3️⃣ Book / Notebook
    # -----------------------------
    nb = objects.get("book")
    reading_posture = False
    if nb is not None:
        if (left_hand is not None and point_to_box(left_hand, nb) < BOOK_DIST) or \
           (right_hand is not None and point_to_box(right_hand, nb) < BOOK_DIST):
            detected_book = True

        if nose is not None and point_to_box(nose, nb) < 200:
            reading_posture = True

    # -----------------------------
    # 4️⃣ Laptop
    # -----------------------------
    if "laptop" in objects:
        lp = objects["laptop"]
        if (left_hand is not None and point_to_box(left_hand, lp) < LAPTOP_DIST) or \
           (right_hand is not None and point_to_box(right_hand, lp) < LAPTOP_DIST):
            detected_laptop = True

    # -----------------------------------------------
    # Update task_memory with small miss-tolerance
    # -----------------------------------------------
    # PHONE
    if detected_phone:
        detection_miss_counters["phone"] = 0
        task_memory["phone"] += 1
    else:
        detection_miss_counters["phone"] += 1
        if detection_miss_counters["phone"] >= DETECT_RESET_FRAMES:
            task_memory["phone"] = 0
            detection_miss_counters["phone"] = 0

    # CUP / BOTTLE
    if detected_cup:
        detection_miss_counters["cup"] = 0
        task_memory["cup"] += 1
    else:
        detection_miss_counters["cup"] += 1
        if detection_miss_counters["cup"] >= DETECT_RESET_FRAMES:
            task_memory["cup"] = 0
            detection_miss_counters["cup"] = 0

    # BOOK
    if detected_book:
        detection_miss_counters["book"] = 0
        task_memory["book"] += 1
    else:
        detection_miss_counters["book"] += 1
        if detection_miss_counters["book"] >= DETECT_RESET_FRAMES:
            task_memory["book"] = 0
            detection_miss_counters["book"] = 0

    # LAPTOP
    if detected_laptop:
        detection_miss_counters["laptop"] = 0
        task_memory["laptop"] += 1
    else:
        detection_miss_counters["laptop"] += 1
        if detection_miss_counters["laptop"] >= DETECT_RESET_FRAMES:
            task_memory["laptop"] = 0
            detection_miss_counters["laptop"] = 0

    # -----------------------------
    # PRIORITY LOGIC with acting detection
    # -----------------------------
    # PHONE
    if task_memory["phone"] >= CONFIRM_FRAMES:
        if is_acting("phone", avg_motion, avg_hand_motion):
            # Do not start/continue timer when acting; show explicit label
            return "Acting"
        if update_activity_timer("phone", True, frame):
            return "⚠ Time limit exceeded"
        return "Using mobile phone"
    else:
        update_activity_timer("phone", False, None)

    # CUP
    if task_memory["cup"] >= CONFIRM_FRAMES:
        if is_acting("cup", avg_motion, avg_hand_motion):
            return "Acting"
        if update_activity_timer("cup", True, frame):
            return "⚠ Time limit exceeded"
        return "Drinking" if drinking else "Holding cup"
    else:
        update_activity_timer("cup", False, None)

    # BOOK
    if task_memory["book"] >= CONFIRM_FRAMES:
        if is_acting("book", avg_motion, avg_hand_motion):
            return "Acting"
        if update_activity_timer("book", True, frame):
            return "⚠ Time limit exceeded"
        return "Reading" if reading_posture else "Holding notebook"
    else:
        update_activity_timer("book", False, None)

    # LAPTOP
    if task_memory["laptop"] >= CONFIRM_FRAMES:
        if is_acting("laptop", avg_motion, avg_hand_motion):
            return "Acting"
        if update_activity_timer("laptop", True, frame):
            return "⚠ Time limit exceeded"
        return "using Laptop"
    else:
        update_activity_timer("laptop", False, None)

    return "Idle"

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose and hands
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    # Extract keypoints
    kps = extract_keypoints(results_pose, results_hands, frame.shape)

    # Draw skeleton mask
    mask = np.zeros_like(frame)
    mask = draw_pose_mask(mask, kps)

    # Motion calculation (body-wide)
    if prev_keypoints is not None:
        # Only consider valid keypoints (non-zero)
        valid = (kps[:,0] > 0) & (prev_keypoints[:,0] > 0)

        if np.any(valid):
            diff = np.linalg.norm(kps[valid] - prev_keypoints[valid], axis=1)
            movement = np.mean(diff)
        else:
            movement = 0

        movement_history.append(movement)

    prev_keypoints = kps.copy()

    if len(movement_history) > STATIC_FRAMES:
        movement_history.pop(0)

    avg_motion = np.mean(movement_history) if movement_history else 0

    # -----------------------------
    # Hand-specific motion
    # -----------------------------
    # left_hand_indices = 33..53, right_hand_indices = 54..74
    if prev_keypoints is not None:
        # compute per-keypoint movement for hands; filter zeros
        left_idx = list(range(33, 54))
        right_idx = list(range(54, 75))

        # ensure prev_keypoints exists earlier than current (we copy above)
        # here prev_keypoints is current frame's previous stored state, but we've already set prev_keypoints = kps.copy()
        # so we need a separate previous variable; to avoid complicating, we compute hand motion using movement_history approach:
        # We'll approximate hand motion from the most recent two stored frames in movement_history by recomputing using last two kps snapshots.
        # Simpler: keep last_kps variable. Let's implement last_kps variable outside loop. (We'll add it dynamically.)

        pass

    # To compute hand motion robustly we need the previous keypoints from the prior frame.
    # We'll implement last_kps to hold the previous frame's keypoints.
    # (Define last_kps if not present.)
    try:
        last_kps
    except NameError:
        last_kps = None

    # compute avg_hand_motion using last_kps (previous frame) and current kps
    if last_kps is not None:
        left_idx = list(range(33, 54))
        right_idx = list(range(54, 75))

        left_valid = (kps[left_idx,0] > 0) & (last_kps[left_idx,0] > 0)
        right_valid = (kps[right_idx,0] > 0) & (last_kps[right_idx,0] > 0)

        if np.any(left_valid):
            left_motion = np.linalg.norm(kps[left_idx][left_valid] - last_kps[left_idx][left_valid], axis=1)
            avg_left_hand_motion = np.mean(left_motion)
        else:
            avg_left_hand_motion = 0.0

        if np.any(right_valid):
            right_motion = np.linalg.norm(kps[right_idx][right_valid] - last_kps[right_idx][right_valid], axis=1)
            avg_right_hand_motion = np.mean(right_motion)
        else:
            avg_right_hand_motion = 0.0

        avg_hand_motion = (avg_left_hand_motion + avg_right_hand_motion) / 2.0
    else:
        avg_hand_motion = 0.0

    # update last_kps for next iteration
    last_kps = kps.copy()

    # -----------------------------
    # YOLO Object Detection
    # -----------------------------
    CONF_THRESH = 0.7
    # include both cup and bottle class names
    SELECTED_CLASSES = ["cell phone", "laptop"]
    obj_results = object_model(frame)[0]
    objects = {}
    for box, cls, conf in zip(obj_results.boxes.xyxy, obj_results.boxes.cls, obj_results.boxes.conf):
        clss = object_model.names[int(cls)]
        conf = float(conf)

        if conf < CONF_THRESH or clss not in SELECTED_CLASSES:
            continue

        label = clss
        objects[label] = box.cpu().numpy()

        # Draw detected objects
        x1, y1, x2, y2 = box.cpu().numpy()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    frame_global = frame.copy()

    # -----------------------------
    # Detect task (pass motion metrics)
    # -----------------------------
    task = detect_task(objects, kps, avg_motion, avg_hand_motion, frame_global)

    # If motion too low, mark as Not Working / Working only when Idle
    if task == "Idle":
        if avg_motion < MOTION_THRESHOLD:
            task = "Not Working"
        else:
            task = "Working"

    # Draw detailed status (you can also show avg motions for debugging)
    status_text = f"Task: {task}"
    debug_text = f"BodyM: {avg_motion:.2f} HandM: {avg_hand_motion:.2f}"
    cv2.putText(frame, status_text, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0),4)
    # cv2.putText(frame, debug_text, (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

    overlay = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)
    cv2.imshow("Task + Pose", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
hands.close()

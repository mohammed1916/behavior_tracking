import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from utils import calculate_distance, is_point_in_box, draw_info_panel, boxes_intersect

# Constants
TASK_TIME_LIMIT = 15.0 # seconds
OPENING_THRESHOLD_FRAMES = 30 # Number of frames in "OPENING" to count as done
IDLE_CAPTURE_THRESHOLD = 3.0 # Seconds of idle before capturing
CAPTURE_DIR = "captured_frames"

class BehaviorTracker:
    def __init__(self):
        # Initialize Models
        print("Initializing YOLOv8...")
        self.yolo_model = YOLO('yolov8n.pt')
        
        print("Initializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State
        self.state = "IDLE" # IDLE, HOLDING, OPENING, COMPLETED
        self.task_start_time = None
        self.task_completed = False
        self.idle_start_time = time.time()
        self.opening_frame_counter = 0
        
        # Setup capture dir
        if not os.path.exists(CAPTURE_DIR):
            os.makedirs(CAPTURE_DIR)
            
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Object Detection (Bottle)
        results = self.yolo_model(frame, verbose=False, classes=[39]) # 39 is bottle in COCO
        bottle_box = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                if conf > 0.4:
                    bottle_box = [int(x1), int(y1), int(x2), int(y2)]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, "Bottle", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    break # Assume one bottle for simplicity
        
        # 2. Hand Tracking
        hand_results = self.hands.process(rgb_frame)
        hands_near_bottle = 0
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check if hand is near bottle
                if bottle_box:
                    # Use wrist as proxy for hand position
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    wrist_px = (int(wrist.x * w), int(wrist.y * h))
                    
                    # Expand bottle box slightly for "near" check
                    bx1, by1, bx2, by2 = bottle_box
                    margin = 50
                    expanded_box = [bx1-margin, by1-margin, bx2+margin, by2+margin]
                    
                    if is_point_in_box(wrist_px, expanded_box):
                        hands_near_bottle += 1
                        
        # 3. State Machine Logic
        current_time = time.time()
        
        if self.task_completed:
            self.state = "COMPLETED"
        elif bottle_box:
            if hands_near_bottle == 0:
                self.state = "IDLE"
                self.task_start_time = None # Reset timer if let go? Or pause? Let's reset for now.
            elif hands_near_bottle == 1:
                self.state = "HOLDING"
                if self.task_start_time is None:
                    self.task_start_time = current_time
            elif hands_near_bottle >= 2:
                self.state = "OPENING"
                if self.task_start_time is None:
                    self.task_start_time = current_time
                
                # Simple "Opening" completion logic: maintain state for N frames
                self.opening_frame_counter += 1
                if self.opening_frame_counter > OPENING_THRESHOLD_FRAMES:
                    self.task_completed = True
            else:
                self.opening_frame_counter = max(0, self.opening_frame_counter - 1)
        else:
            self.state = "IDLE"
            self.task_start_time = None
            self.opening_frame_counter = 0

        # 4. Timer Logic
        time_left = None
        if self.task_start_time and not self.task_completed:
            elapsed = current_time - self.task_start_time
            time_left = max(0, TASK_TIME_LIMIT - elapsed)
            if time_left == 0:
                # Time out logic could go here
                pass

        # 5. Idle Capture Logic
        if self.state == "IDLE":
            if current_time - self.idle_start_time > IDLE_CAPTURE_THRESHOLD:
                # Capture frame
                filename = os.path.join(CAPTURE_DIR, f"idle_{int(current_time)}.jpg")
                # Only capture once every few seconds to avoid spam
                if int(current_time) % 2 == 0: 
                     # Check if we haven't already captured this second (simple debounce)
                     if not os.path.exists(filename):
                        cv2.imwrite(filename, frame)
                        # print(f"Captured idle frame: {filename}") # Reduce noise in logs
        else:
            self.idle_start_time = current_time

        # Draw UI
        draw_info_panel(frame, self.state, time_left, self.task_completed)
        
        return frame, self.state, self.task_completed

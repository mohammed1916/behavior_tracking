import os
"""Configuration constants for activity tracking system"""

# Model configuration
VLM_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Motion detection thresholds
MOTION_HISTORY_SIZE = 10
WORK_MOTION_MIN = 2.0          # Minimum motion in work area
WORK_MOTION_MAX = 15.0         # Maximum (too much = not focused)
HAND_SCORE_MIN = 2             # Minimum hand-like motion regions
WORK_RATIO_MIN = 1.2           # Work area should have more motion than upper body
MOTION_REGION_MIN_AREA = 50    # Minimum contour area to consider
HAND_MOTION_MIN_AREA = 100     # Minimum area for hand-like motion
HAND_MOTION_MAX_AREA = 5000    # Maximum area for hand-like motion
MOTION_THRESHOLD = 25          # Threshold for motion detection

# MediaPipe configuration
MEDIAPIPE_HANDS_CONFIDENCE = 0.5
MEDIAPIPE_POSE_CONFIDENCE = 0.5

# Activity tracking limits (seconds)
IDLE_LIMIT = 10
PHONE_LIMIT = 10
DRONE_LIMIT = 20

# Video properties
VIDEO_FPS = 30
VLM_CLASSIFY_INTERVAL = 1.0    # Classify activity once per second (seconds)
PHONE_RESET_FRAMES = 5          # Frames to wait before resetting phone detection

# Throughput and scaling
TARGET_PROCESS_FPS = 24.0        # Process only N frames per second (ingest may be higher)
PROCESSING_DOWNSCALE = 0.5      # Resize factor for inference to speed up processing
OUTPUT_FPS = TARGET_PROCESS_FPS # Save output video at the processed frame rate

# Output
CSV_FILE = os.path.join("logs", "activity_log.csv")
VIDEO_OUTPUT_FILE = os.path.join("logs", "data_collection_.mp4")
VIDEO_CODEC = 'mp4v'

# Activity labels
ACTIVITY_ASSEMBLING_DRONE = "assembling_drone"
ACTIVITY_IDLE = "idle"
ACTIVITY_USING_PHONE = "using_phone"
ACTIVITY_SIMPLY_SITTING = "simply_sitting"
ACTIVITY_UNKNOWN = "unknown"

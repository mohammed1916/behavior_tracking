"""MediaPipe-based motion and pose detection for activity recognition"""

import cv2
import numpy as np
import mediapipe as mp
from config import (
    MEDIAPIPE_HANDS_CONFIDENCE, MEDIAPIPE_POSE_CONFIDENCE,
    MOTION_HISTORY_SIZE, WORK_MOTION_MIN, WORK_MOTION_MAX,
    HAND_SCORE_MIN, WORK_RATIO_MIN, MOTION_REGION_MIN_AREA,
    HAND_MOTION_MIN_AREA, HAND_MOTION_MAX_AREA, MOTION_THRESHOLD
)


class MediaPipeMotionDetector:
    """Detects motion and pose using MediaPipe hand and pose tracking"""
    
    def __init__(self):
        """Initialize MediaPipe Hand and Pose detectors"""
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MEDIAPIPE_HANDS_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=MEDIAPIPE_POSE_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        
        self.prev_frame_gray = None
        self.motion_history = []
        self.prev_hand_positions = []
        self.prev_pose_landmarks = None
    
    def detect_hand_motion(self, frame, hand_results):
        """Detect hand motion from MediaPipe hand landmarks"""
        hand_motion_regions = 0
        hand_velocity = 0.0
        
        if hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Calculate hand velocity from position changes
            current_positions = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get hand center
                hand_x = np.mean([lm.x for lm in hand_landmarks.landmark])
                hand_y = np.mean([lm.y for lm in hand_landmarks.landmark])
                current_positions.append((hand_x, hand_y))
                
                # Check if hand is in work area (bottom half)
                if hand_y > 0.5:
                    hand_motion_regions += 1
            
            # Calculate motion velocity
            if self.prev_hand_positions and len(current_positions) == len(self.prev_hand_positions):
                velocities = []
                for curr, prev in zip(current_positions, self.prev_hand_positions):
                    dx = (curr[0] - prev[0]) * w
                    dy = (curr[1] - prev[1]) * h
                    vel = np.sqrt(dx**2 + dy**2)
                    velocities.append(vel)
                hand_velocity = np.mean(velocities) if velocities else 0.0
            
            self.prev_hand_positions = current_positions
        
        return hand_motion_regions, hand_velocity
    
    def detect_pose_activity(self, pose_results):
        """Determine activity type from pose landmarks"""
        if not pose_results.pose_landmarks:
            return None
        
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get key points
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        # Check if hands are near face (phone usage indicator)
        wrist_near_face = False
        if (left_wrist.y < left_ear.y and left_wrist.presence > 0.5) or \
           (right_wrist.y < right_ear.y and right_wrist.presence > 0.5):
            wrist_near_face = True
        
        return {
            'wrists_near_face': wrist_near_face,
            'left_wrist_y': left_wrist.y,
            'right_wrist_y': right_wrist.y,
        }
    
    def analyze_motion_pattern(self, frame_gray, prev_gray):
        """Enhanced motion analysis with region-based detection"""
        h, w = frame_gray.shape
        
        # Calculate overall motion
        diff = cv2.absdiff(frame_gray, prev_gray)
        overall_motion = np.sum(diff) / (h * w)
        
        # Region-based motion analysis (work area detection)
        work_area = diff[h//2:, :]
        upper_area = diff[:h//2, :]
        
        work_motion = np.sum(work_area) / work_area.size if work_area.size > 0 else 0
        upper_motion = np.sum(upper_area) / upper_area.size if upper_area.size > 0 else 0
        
        # Detect localized motion (small, precise movements = productive work)
        _, motion_mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze motion characteristics
        num_motion_regions = len([c for c in contours if cv2.contourArea(c) > MOTION_REGION_MIN_AREA])
        
        # Hand-like motion detection
        hand_motion_regions = [c for c in contours 
                              if HAND_MOTION_MIN_AREA < cv2.contourArea(c) < HAND_MOTION_MAX_AREA]
        hand_motion_score = len(hand_motion_regions)
        
        return {
            'overall': overall_motion,
            'work_area': work_motion,
            'upper_area': upper_motion,
            'num_regions': num_motion_regions,
            'hand_score': hand_motion_score,
            'work_ratio': work_motion / (upper_motion + 0.1)  # Avoid division by zero
        }
    
    def classify_motion_as_productive(self, motion_stats):
        """Determine if motion pattern indicates productive work"""
        self.motion_history.append(motion_stats)
        if len(self.motion_history) > MOTION_HISTORY_SIZE:
            self.motion_history.pop(0)
        
        # Analyze recent motion pattern (smoothing)
        if len(self.motion_history) >= 3:
            avg_work_motion = np.mean([m['work_area'] for m in self.motion_history[-5:]])
            avg_hand_score = np.mean([m['hand_score'] for m in self.motion_history[-5:]])
            avg_work_ratio = np.mean([m['work_ratio'] for m in self.motion_history[-5:]])
            consistency = np.std([m['work_area'] for m in self.motion_history[-5:]])
            
            is_productive = (
                WORK_MOTION_MIN < avg_work_motion < WORK_MOTION_MAX and
                avg_hand_score >= HAND_SCORE_MIN and
                avg_work_ratio >= WORK_RATIO_MIN and
                consistency < 5.0
            )
            
            return is_productive, avg_work_motion, avg_hand_score
        
        return False, 0, 0
    
    def process_frame(self, frame):
        """Process frame and return motion analysis and pose info"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get hand and pose results
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        
        # Detect hand motion
        hand_motion_regions, hand_velocity = self.detect_hand_motion(frame, hand_results)
        
        # Detect pose activity
        pose_info = self.detect_pose_activity(pose_results)
        
        # Analyze optical flow motion
        motion_stats = None
        is_productive_motion = False
        avg_work_motion = 0
        avg_hand_score = 0
        
        if self.prev_frame_gray is not None:
            motion_stats = self.analyze_motion_pattern(frame_gray, self.prev_frame_gray)
            is_productive_motion, avg_work_motion, avg_hand_score = self.classify_motion_as_productive(motion_stats)
        
        self.prev_frame_gray = frame_gray
        
        return {
            'hand_results': hand_results,
            'pose_results': pose_results,
            'pose_info': pose_info,
            'hand_motion_regions': hand_motion_regions,
            'hand_velocity': hand_velocity,
            'motion_stats': motion_stats,
            'is_productive_motion': is_productive_motion,
            'avg_work_motion': avg_work_motion,
            'avg_hand_score': avg_hand_score
        }
    
    def draw_landmarks(self, frame, hand_results, pose_results):
        """Draw MediaPipe landmarks on frame for visualization"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_rgb,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    def release(self):
        """Release MediaPipe resources"""
        self.hands.close()
        self.pose.close()

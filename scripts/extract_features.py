"""Extract per-frame features from video using MediaPipe + YOLO for task classification.

Outputs a Parquet file with columns:
- video_id: unique video identifier
- frame_index: 0-based frame number
- t: timestamp in seconds
- mp_pose_landmarks: 33x3 pose landmarks (x, y, z normalized)
- mp_left_hand_landmarks: 21x3 left hand landmarks
- mp_right_hand_landmarks: 21x3 right hand landmarks
- yolo_objects: list of detected object classes
- yolo_boxes: list of bounding boxes [x, y, w, h, conf] per object
- derived_features: dict with angles, velocities, hand-object distances, etc.
- label: optional ground-truth label (if annotations provided)

Usage:
    python scripts/extract_features.py --video path/to/video.mp4 --output features.parquet
    python scripts/extract_features.py --video path/to/video.mp4 --labels labels.csv --output features.parquet
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.detectors.mediapipe_yolo_detector import MediaPipeYoloDetector
from backend.detectors.mediapipe_detector import MediaPipeDetector
from backend.detectors.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_derived_features(mp_output: Optional[Dict], yolo_output: Optional[Dict], prev_features: Optional[Dict] = None) -> Dict:
    """Compute derived features from raw detector outputs.
    
    Args:
        mp_output: MediaPipe metadata dict with pose/hand landmarks
        yolo_output: YOLO metadata dict with detections
        prev_features: Previous frame's derived features for velocity computation
    
    Returns:
        Dict with derived features: angles, velocities, distances, etc.
    """
    features = {}
    
    # MediaPipe-derived features
    if mp_output:
        # Pose angles (elbow, shoulder, knee, hip)
        pose = mp_output.get('pose_landmarks', [])
        if len(pose) >= 33:  # Full pose detected
            # Right arm angle (shoulder-elbow-wrist: landmarks 12-14-16)
            try:
                shoulder = np.array([pose[12]['x'], pose[12]['y'], pose[12]['z']])
                elbow = np.array([pose[14]['x'], pose[14]['y'], pose[14]['z']])
                wrist = np.array([pose[16]['x'], pose[16]['y'], pose[16]['z']])
                
                v1 = shoulder - elbow
                v2 = wrist - elbow
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0))
                features['right_elbow_angle'] = float(np.degrees(angle))
            except Exception:
                features['right_elbow_angle'] = None
            
            # Left arm angle (11-13-15)
            try:
                shoulder = np.array([pose[11]['x'], pose[11]['y'], pose[11]['z']])
                elbow = np.array([pose[13]['x'], pose[13]['y'], pose[13]['z']])
                wrist = np.array([pose[15]['x'], pose[15]['y'], pose[15]['z']])
                
                v1 = shoulder - elbow
                v2 = wrist - elbow
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0))
                features['left_elbow_angle'] = float(np.degrees(angle))
            except Exception:
                features['left_elbow_angle'] = None
            
            # Hand positions (center of mass)
            right_hand = mp_output.get('right_hand_landmarks', [])
            left_hand = mp_output.get('left_hand_landmarks', [])
            
            if right_hand:
                features['right_hand_x'] = float(np.mean([lm['x'] for lm in right_hand]))
                features['right_hand_y'] = float(np.mean([lm['y'] for lm in right_hand]))
            else:
                features['right_hand_x'] = None
                features['right_hand_y'] = None
            
            if left_hand:
                features['left_hand_x'] = float(np.mean([lm['x'] for lm in left_hand]))
                features['left_hand_y'] = float(np.mean([lm['y'] for lm in left_hand]))
            else:
                features['left_hand_x'] = None
                features['left_hand_y'] = None
            
            # Hand velocities (if previous frame available)
            if prev_features:
                if features['right_hand_x'] is not None and prev_features.get('right_hand_x') is not None:
                    features['right_hand_vx'] = features['right_hand_x'] - prev_features['right_hand_x']
                    features['right_hand_vy'] = features['right_hand_y'] - prev_features['right_hand_y']
                    features['right_hand_speed'] = np.sqrt(features['right_hand_vx']**2 + features['right_hand_vy']**2)
                else:
                    features['right_hand_vx'] = 0.0
                    features['right_hand_vy'] = 0.0
                    features['right_hand_speed'] = 0.0
                
                if features['left_hand_x'] is not None and prev_features.get('left_hand_x') is not None:
                    features['left_hand_vx'] = features['left_hand_x'] - prev_features['left_hand_x']
                    features['left_hand_vy'] = features['left_hand_y'] - prev_features['left_hand_y']
                    features['left_hand_speed'] = np.sqrt(features['left_hand_vx']**2 + features['left_hand_vy']**2)
                else:
                    features['left_hand_vx'] = 0.0
                    features['left_hand_vy'] = 0.0
                    features['left_hand_speed'] = 0.0
            else:
                features['right_hand_vx'] = 0.0
                features['right_hand_vy'] = 0.0
                features['right_hand_speed'] = 0.0
                features['left_hand_vx'] = 0.0
                features['left_hand_vy'] = 0.0
                features['left_hand_speed'] = 0.0
    
    # YOLO-derived features
    if yolo_output:
        detections = yolo_output.get('detections', [])
        features['num_objects'] = len(detections)
        
        # Count by category
        class_counts = {}
        for det in detections:
            cls = det.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        features['class_counts'] = class_counts
        
        # Distances from hands to nearest objects (if both MP and YOLO available)
        if mp_output and detections:
            # Right hand to nearest object
            right_hand_x = features.get('right_hand_x')
            right_hand_y = features.get('right_hand_y')
            
            if right_hand_x is not None:
                min_dist = float('inf')
                for det in detections:
                    box = det.get('bbox', {})
                    obj_x = box.get('x', 0) + box.get('w', 0) / 2
                    obj_y = box.get('y', 0) + box.get('h', 0) / 2
                    dist = np.sqrt((right_hand_x - obj_x)**2 + (right_hand_y - obj_y)**2)
                    min_dist = min(min_dist, dist)
                features['right_hand_obj_dist'] = float(min_dist) if min_dist != float('inf') else None
            else:
                features['right_hand_obj_dist'] = None
            
            # Left hand to nearest object
            left_hand_x = features.get('left_hand_x')
            left_hand_y = features.get('left_hand_y')
            
            if left_hand_x is not None:
                min_dist = float('inf')
                for det in detections:
                    box = det.get('bbox', {})
                    obj_x = box.get('x', 0) + box.get('w', 0) / 2
                    obj_y = box.get('y', 0) + box.get('h', 0) / 2
                    dist = np.sqrt((left_hand_x - obj_x)**2 + (left_hand_y - obj_y)**2)
                    min_dist = min(min_dist, dist)
                features['left_hand_obj_dist'] = float(min_dist) if min_dist != float('inf') else None
            else:
                features['left_hand_obj_dist'] = None
    
    return features


def extract_features_from_video(
    video_path: str,
    output_path: str,
    video_id: Optional[str] = None,
    labels_csv: Optional[str] = None,
    detector_type: str = 'fusion',
    sample_rate: int = 1,
    device: str = 'cuda:0',
) -> None:
    """Extract features from video and save to Parquet.
    
    Args:
        video_path: Path to input video
        output_path: Path to output Parquet file
        video_id: Unique video identifier (default: filename)
        labels_csv: Optional CSV with frame-level labels (columns: frame_index, label)
        detector_type: 'fusion' (MP+YOLO), 'mediapipe', or 'yolo'
        sample_rate: Process every Nth frame (1 = all frames)
        device: Device for YOLO ('cuda:0' or 'cpu')
    """
    if video_id is None:
        video_id = Path(video_path).stem
    
    logger.info(f"Extracting features from {video_path}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Detector: {detector_type}")
    logger.info(f"Sample rate: 1/{sample_rate}")
    
    # Load labels if provided
    labels_map = {}
    if labels_csv and os.path.exists(labels_csv):
        df_labels = pd.read_csv(labels_csv)
        labels_map = dict(zip(df_labels['frame_index'], df_labels['label']))
        logger.info(f"Loaded {len(labels_map)} labels from {labels_csv}")
    
    # Initialize detector
    if detector_type == 'fusion':
        detector = MediaPipeYoloDetector(
            yolo_model='yolov8n',
            mediapipe_confidence=0.5,
            yolo_confidence=0.5,
            strategy='cascade',
            device=device,
        )
    elif detector_type == 'mediapipe':
        detector = MediaPipeDetector(confidence_threshold=0.5)
    elif detector_type == 'yolo':
        detector = YOLODetector(model='yolov8n', confidence_threshold=0.5, device=device)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {frame_count} frames @ {fps:.2f} fps")
    
    # Extract features frame by frame
    rows = []
    frame_idx = 0
    prev_derived = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample rate filtering
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            t = frame_idx / fps
            
            # Run detector
            try:
                output = detector.process_frame(frame)
                metadata = output.metadata
            except Exception as e:
                logger.warning(f"Frame {frame_idx}: detector failed - {e}")
                metadata = {}
            
            # Extract raw features
            mp_output = metadata.get('mediapipe', {}) if detector_type in ['fusion', 'mediapipe'] else None
            yolo_output = metadata.get('yolo', {}) if detector_type in ['fusion', 'yolo'] else None
            
            # Compute derived features
            derived = compute_derived_features(mp_output, yolo_output, prev_derived)
            prev_derived = derived
            
            # Build row
            row = {
                'video_id': video_id,
                'frame_index': frame_idx,
                't': t,
                'label': labels_map.get(frame_idx, None),
            }
            
            # MediaPipe features (store as JSON strings for Parquet compatibility)
            if mp_output:
                row['mp_pose_landmarks'] = json.dumps(mp_output.get('pose_landmarks', []))
                row['mp_left_hand_landmarks'] = json.dumps(mp_output.get('left_hand_landmarks', []))
                row['mp_right_hand_landmarks'] = json.dumps(mp_output.get('right_hand_landmarks', []))
            else:
                row['mp_pose_landmarks'] = None
                row['mp_left_hand_landmarks'] = None
                row['mp_right_hand_landmarks'] = None
            
            # YOLO features
            if yolo_output:
                detections = yolo_output.get('detections', [])
                row['yolo_objects'] = json.dumps([d.get('class') for d in detections])
                row['yolo_boxes'] = json.dumps([d.get('bbox') for d in detections])
                row['yolo_confidences'] = json.dumps([d.get('confidence') for d in detections])
            else:
                row['yolo_objects'] = None
                row['yolo_boxes'] = None
                row['yolo_confidences'] = None
            
            # Derived features (flatten dict into columns)
            for k, v in derived.items():
                if k == 'class_counts':
                    row['class_counts'] = json.dumps(v)
                else:
                    row[k] = v
            
            rows.append(row)
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames ({100*frame_idx/frame_count:.1f}%)")
            
            frame_idx += 1
    
    finally:
        cap.release()
        detector.close()
    
    # Save to Parquet
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df)} feature rows to {output_path}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Print summary stats
    if 'label' in df.columns and df['label'].notna().any():
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from video for task classification')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', required=True, help='Path to output Parquet file')
    parser.add_argument('--video-id', help='Unique video identifier (default: filename)')
    parser.add_argument('--labels', help='CSV file with frame labels (columns: frame_index, label)')
    parser.add_argument('--detector', choices=['fusion', 'mediapipe', 'yolo'], default='fusion',
                        help='Detector type (default: fusion)')
    parser.add_argument('--sample-rate', type=int, default=1, help='Process every Nth frame (default: 1)')
    parser.add_argument('--device', default='cuda:0', help='Device for YOLO (default: cuda:0)')
    
    args = parser.parse_args()
    
    extract_features_from_video(
        video_path=args.video,
        output_path=args.output,
        video_id=args.video_id,
        labels_csv=args.labels,
        detector_type=args.detector,
        sample_rate=args.sample_rate,
        device=args.device,
    )


if __name__ == '__main__':
    main()

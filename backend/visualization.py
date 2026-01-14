"""Visualization utilities for detector overlays on video frames.

Provides functions to draw YOLO bounding boxes and MediaPipe keypoints/skeletons
on video frames for debugging and demo purposes.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


# MediaPipe pose connections (standard skeleton)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # Right leg
]

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]


def draw_yolo_detections(
    frame: np.ndarray,
    detections: Dict[str, List[Dict[str, Any]]],
    confidence_threshold: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_labels: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """Draw YOLO bounding boxes on frame.
    
    Args:
        frame: OpenCV frame (BGR numpy array)
        detections: Dict mapping class names to list of detections
            Each detection: {'confidence': float, 'bbox': [x1, y1, x2, y2]}
        confidence_threshold: Optional threshold to filter detections
        color: BGR color tuple for boxes
        thickness: Line thickness for boxes
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
    
    Returns:
        Annotated frame
    """
    frame_annotated = frame.copy()
    h, w = frame.shape[:2]
    
    for class_name, detection_list in detections.items():
        for detection in detection_list:
            conf = detection.get('confidence', 0.0)
            
            # Skip if below threshold
            if confidence_threshold and conf < confidence_threshold:
                continue
            
            bbox = detection.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Draw bounding box
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_confidence:
                    label_parts.append(f"{conf:.2f}")
                label = " ".join(label_parts)
                
                # Background for text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame_annotated,
                    (x1, y1 - text_h - baseline - 5),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    frame_annotated,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )
    
    return frame_annotated


def draw_mediapipe_landmarks(
    frame: np.ndarray,
    hand_landmarks: Optional[List[List[Dict[str, float]]]] = None,
    pose_landmarks: Optional[List[Dict[str, float]]] = None,
    hand_color: Tuple[int, int, int] = (255, 0, 0),
    pose_color: Tuple[int, int, int] = (0, 255, 255),
    landmark_radius: int = 3,
    connection_thickness: int = 2,
    min_visibility: float = 0.5
) -> np.ndarray:
    """Draw MediaPipe hand and pose landmarks on frame.
    
    Args:
        frame: OpenCV frame (BGR numpy array)
        hand_landmarks: List of hand landmark lists
            Each hand: list of 21 landmarks with {'x', 'y', 'z'}
        pose_landmarks: List of pose landmarks
            List of 33 landmarks with {'x', 'y', 'z', 'visibility'}
        hand_color: BGR color for hand landmarks
        pose_color: BGR color for pose landmarks
        landmark_radius: Radius of landmark circles
        connection_thickness: Thickness of skeleton connections
        min_visibility: Minimum visibility threshold for pose landmarks
    
    Returns:
        Annotated frame
    """
    frame_annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw hand landmarks
    if hand_landmarks:
        for hand in hand_landmarks:
            if not hand or len(hand) < 21:
                continue
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx >= len(hand) or end_idx >= len(hand):
                    continue
                
                start_lm = hand[start_idx]
                end_lm = hand[end_idx]
                
                start_x = int(start_lm['x'] * w)
                start_y = int(start_lm['y'] * h)
                end_x = int(end_lm['x'] * w)
                end_y = int(end_lm['y'] * h)
                
                cv2.line(
                    frame_annotated,
                    (start_x, start_y),
                    (end_x, end_y),
                    hand_color,
                    connection_thickness
                )
            
            # Draw landmarks
            for lm in hand:
                x = int(lm['x'] * w)
                y = int(lm['y'] * h)
                cv2.circle(frame_annotated, (x, y), landmark_radius, hand_color, -1)
    
    # Draw pose landmarks
    if pose_landmarks and len(pose_landmarks) >= 33:
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                continue
            
            start_lm = pose_landmarks[start_idx]
            end_lm = pose_landmarks[end_idx]
            
            # Check visibility
            start_vis = start_lm.get('visibility', 1.0)
            end_vis = end_lm.get('visibility', 1.0)
            if start_vis < min_visibility or end_vis < min_visibility:
                continue
            
            start_x = int(start_lm['x'] * w)
            start_y = int(start_lm['y'] * h)
            end_x = int(end_lm['x'] * w)
            end_y = int(end_lm['y'] * h)
            
            cv2.line(
                frame_annotated,
                (start_x, start_y),
                (end_x, end_y),
                pose_color,
                connection_thickness
            )
        
        # Draw landmarks
        for lm in pose_landmarks:
            vis = lm.get('visibility', 1.0)
            if vis < min_visibility:
                continue
            
            x = int(lm['x'] * w)
            y = int(lm['y'] * h)
            cv2.circle(frame_annotated, (x, y), landmark_radius, pose_color, -1)
    
    return frame_annotated


def draw_detector_overlay(
    frame: np.ndarray,
    detector_metadata: Dict[str, Any],
    show_yolo: bool = True,
    show_mediapipe: bool = True,
    show_info: bool = True
) -> np.ndarray:
    """Draw complete detector overlay on frame.
    
    Args:
        frame: OpenCV frame (BGR numpy array)
        detector_metadata: Detector metadata dict from sample
        show_yolo: Whether to show YOLO detections
        show_mediapipe: Whether to show MediaPipe landmarks
        show_info: Whether to show info text overlay
    
    Returns:
        Annotated frame
    """
    if not detector_metadata:
        return frame
    
    frame_annotated = frame.copy()
    
    # Draw YOLO detections
    if show_yolo and 'detections' in detector_metadata:
        detections = detector_metadata['detections']
        confidence_threshold = detector_metadata.get('confidence_threshold', 0.5)
        frame_annotated = draw_yolo_detections(
            frame_annotated,
            detections,
            confidence_threshold=confidence_threshold
        )
    
    # Draw MediaPipe landmarks
    if show_mediapipe:
        hand_landmarks = detector_metadata.get('hand_landmarks')
        pose_landmarks = detector_metadata.get('pose_landmarks')
        frame_annotated = draw_mediapipe_landmarks(
            frame_annotated,
            hand_landmarks=hand_landmarks,
            pose_landmarks=pose_landmarks
        )
    
    # Draw info text
    if show_info:
        info_lines = []
        
        detector_label = detector_metadata.get('detector_label')
        detector_conf = detector_metadata.get('detector_confidence')
        if detector_label:
            info_lines.append(f"Label: {detector_label}")
        if detector_conf is not None:
            info_lines.append(f"Conf: {detector_conf:.2f}")
        
        # YOLO info
        if 'detections' in detector_metadata:
            total_objects = sum(
                len(v) for v in detector_metadata['detections'].values()
            )
            info_lines.append(f"YOLO: {total_objects} objects")
        
        # MediaPipe info
        hand_vel = detector_metadata.get('hand_velocity')
        if hand_vel is not None:
            info_lines.append(f"Hand vel: {hand_vel:.1f} px/frame")
        
        # Draw text overlay
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame_annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame_annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 25
    
    return frame_annotated


def create_annotated_video(
    input_video_path: str,
    output_video_path: str,
    samples: List[Dict[str, Any]],
    show_yolo: bool = True,
    show_mediapipe: bool = True,
    show_info: bool = True,
    progress_callback: Optional[Any] = None
) -> bool:
    """Create annotated video with detector overlays.
    
    Args:
        input_video_path: Path to source video
        output_video_path: Path to output video
        samples: List of samples with detector_metadata
        show_yolo: Whether to show YOLO overlays
        show_mediapipe: Whether to show MediaPipe overlays
        show_info: Whether to show info text
        progress_callback: Optional callback(frame_idx, total_frames)
    
    Returns:
        True if successful
    """
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        # browsers typically prefer avc1 (H.264) over mp4v for .mp4, or vp8/vp9 for .webm
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                # Fallback to mp4v if avc1 fails (though browser might not like it, at least file is created)
                logger.warning("avc1 codec failed, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Index samples by frame index
        samples_by_frame = {}
        for sample in samples:
            frame_idx = sample.get('frame_index')
            if frame_idx is not None:
                samples_by_frame[frame_idx] = sample
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply overlay if sample exists
            if frame_idx in samples_by_frame:
                sample = samples_by_frame[frame_idx]
                detector_metadata = sample.get('detector_metadata')
                if detector_metadata:
                    frame = draw_detector_overlay(
                        frame,
                        detector_metadata,
                        show_yolo=show_yolo,
                        show_mediapipe=show_mediapipe,
                        show_info=show_info
                    )
            
            out.write(frame)
            frame_idx += 1
            
            if progress_callback:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        out.release()
        
        logger.info(f"Created annotated video: {output_video_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error creating annotated video: {e}")
        return False

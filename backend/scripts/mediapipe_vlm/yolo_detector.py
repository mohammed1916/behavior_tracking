"""YOLO-based object detection for activity recognition"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


class YOLOObjectDetector:
    """Detects objects using YOLOv8 for activity context"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt or yolov8l.pt)
        """
        # Check if model exists in parent directory
        if not os.path.exists(model_path):
            # Try parent directory
            parent_model = os.path.join("..", model_path)
            if os.path.exists(parent_model):
                model_path = parent_model
        
        self.model = YOLO(model_path)
        self.device = self.model.device
        
        # Common object classes relevant to activity detection
        self.relevant_classes = {
            'person': 0,
            'phone': 67,
            'laptop': 73,
            'cup': 47,
            'book': 84,
            'tools': [36, 37, 38, 39, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        }
    
    def detect_objects(self, frame, conf_threshold=0.5):
        """Detect objects in frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detections
        
        Returns:
            dict: Detection results with object counts and positions
        """
        # Run YOLO detection
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = {
            'objects': {},
            'object_boxes': [],
            'raw_results': results
        }
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                
                class_name = result.names[cls]
                
                # Store detection
                if class_name not in detections['objects']:
                    detections['objects'][class_name] = {
                        'count': 0,
                        'boxes': [],
                        'confidences': []
                    }
                
                detections['objects'][class_name]['count'] += 1
                detections['objects'][class_name]['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
                detections['objects'][class_name]['confidences'].append(conf)
                
                detections['object_boxes'].append({
                    'class': class_name,
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        return detections
    
    def draw_detections(self, frame, detections, thickness=2):
        """Draw YOLO detections on frame
        
        Args:
            frame: Input frame
            detections: Detection dictionary from detect_objects()
            thickness: Line thickness
        
        Returns:
            frame: Annotated frame
        """
        colors = {
            'phone': (0, 0, 255),      # Red
            'laptop': (255, 0, 0),     # Blue
            'cup': (0, 255, 255),      # Yellow
            'book': (255, 255, 0),     # Cyan
            'person': (0, 255, 0),     # Green
            'tools': (255, 0, 255)     # Magenta
        }
        
        for obj_box in detections['object_boxes']:
            class_name = obj_box['class']
            x1, y1, x2, y2 = obj_box['box']
            conf = obj_box['confidence']
            
            color = colors.get(class_name, (200, 200, 200))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def release(self):
        """Release YOLO resources"""
        # YOLO cleanup if needed
        pass

"""YOLO-based object detector for activity context"""

import cv2
import os
import logging
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from base import DetectorBase, DetectorOutput

logger = logging.getLogger(__name__)


class YOLODetector(DetectorBase):
    """Activity detection via object and tool detection with YOLO"""
    
    # Objects that indicate work activity
    WORK_INDICATORS = {
        'laptop', 'keyboard', 'mouse', 'phone', 'book', 
        'pen', 'pencil', 'tool', 'wrench', 'hammer',
        'screwdriver', 'scissors', 'cup', 'bottle'
    }
    
    # Objects that indicate idle/no work
    IDLE_INDICATORS = {
        'person',  # Just presence of person without tools
    }
    
    def __init__(
        self,
        model: str = 'yolov8n',
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(name='yolo', confidence_threshold=confidence_threshold)
        
        if not HAS_YOLO:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        # Resolve model path
        model_path = self._resolve_model_path(model)
        
        try:
            self.model = YOLO(model_path)
            if device:
                self.model.to(device)
            self.device = self.model.device
            logger.info(f"YOLO loaded: {model_path} on device {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_path}: {e}")
            raise
        
        self.frame_count = 0
        self.last_detections = {}
    
    @staticmethod
    def _resolve_model_path(model: str) -> str:
        """Resolve model name/path, checking multiple locations"""
        candidates = [
            model,
            f"{model}.pt",
            os.path.join("..", model),
            os.path.join("..", f"{model}.pt"),
            os.path.expanduser("~/.cache/yolov8/"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        # If not found, return name and let YOLO download it
        return f"{model}.pt" if not model.endswith('.pt') else model
    
    def process_frame(self, frame) -> DetectorOutput:
        """
        Analyze frame for context (presence of tools/objects).
        
        NOTE: YOLO provides CONTEXT ONLY, not primary activity label.
        The fusion engine uses this as supporting signal:
        - Motion (MediaPipe) = primary signal
        - Tool presence (YOLO) = context/confirmation signal
        
        Always returns 'context' label - fusion engine makes final decision.
        Metadata contains detected work objects for cascade decision tree.
        """
        self.frame_count += 1
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            if not results:
                return DetectorOutput(
                    label='context',
                    confidence=0.0,
                    metadata={
                        'error': 'No results from YOLO',
                        'work_objects_detected': False,
                        'detected_objects': {}
                    }
                )
            
            result = results[0]
            
            # Parse detections
            detected_objects: Dict[str, list] = {}
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id].lower()
                
                if class_name not in detected_objects:
                    detected_objects[class_name] = []
                detected_objects[class_name].append({
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
            
            # Cache for temporal consistency
            self.last_detections = detected_objects
            
            # Check for work indicators (tools/equipment in frame)
            work_objects = set(detected_objects.keys()) & self.WORK_INDICATORS
            has_work_tools = len(work_objects) > 0
            
            # Calculate confidence of tool presence
            tool_confidence = 0.0
            if has_work_tools:
                work_confs = []
                for obj in work_objects:
                    work_confs.extend([d['confidence'] for d in detected_objects[obj]])
                tool_confidence = sum(work_confs) / len(work_confs) if work_confs else 0.5
            
            metadata = {
                'work_objects_detected': has_work_tools,
                'work_objects': list(work_objects),
                'work_object_count': len(work_objects),
                'detected_objects': {k: len(v) for k, v in detected_objects.items()},
                'tool_confidence': tool_confidence,
                'total_detections': len(result.boxes),
                'device': str(self.device),
                'model': self.model.model_name if hasattr(self.model, 'model_name') else 'yolo',
            }
            
            # Return context signal only - fusion engine decides work/idle
            return DetectorOutput(
                label='context',
                confidence=tool_confidence,
                metadata=metadata,
                raw_output={'boxes': result.boxes, 'names': result.names}
            )
        
        except Exception as e:
            logger.warning(f"YOLO detection error: {e}")
            return DetectorOutput(
                label='context',
                confidence=0.0,
                metadata={
                    'error': str(e),
                    'work_objects_detected': False,
                    'detected_objects': {}
                }
            )
    
    def close(self):
        """Clean up YOLO resources"""
        try:
            if self.model:
                # YOLO doesn't have explicit close, but clear GPU cache
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error closing YOLO: {e}")

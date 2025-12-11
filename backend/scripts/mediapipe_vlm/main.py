"""Main activity tracking system orchestrator with YOLO object detection"""

import time
import cv2
import os
# Protobuf compatibility shim: ensure MessageFactory.GetPrototype exists when needed
from protobuf_compat import _mf  # noqa: F401

from motion_detector import MediaPipeMotionDetector
from vlm_classifier import VLMActivityClassifier
from yolo_detector import YOLOObjectDetector
from activity_logger import ActivityLogger, ActivityTracker
from decision_engine import ActivityDecisionEngine
from gpu_monitor import log_gpu_usage
from video_handler import VideoCapture, FrameRenderer
from config import (
    VLM_CLASSIFY_INTERVAL, PHONE_RESET_FRAMES, TARGET_PROCESS_FPS,
    PROCESSING_DOWNSCALE, OUTPUT_FPS, VLM_USE_TWO_MODELS, VLM_ALTERNATE_HALF_INTERVAL,
    ACTIVITY_UNKNOWN
)


class ActivityTrackingSystem:
    """Main system that orchestrates activity tracking with object detection"""
    
    def __init__(self, video_source=0, enable_yolo=True):
        """Initialize the activity tracking system
        
        Args:
            video_source: Video source (0 for webcam, or file path)
            enable_yolo: Whether to enable YOLO object detection
        """
        print("Initializing Activity Tracking System...")
        
        # Initialize components
        self.motion_detector = MediaPipeMotionDetector()
        # VLM classifiers: support single or dual-model alternating mode
        self.vlm_use_two_models = VLM_USE_TWO_MODELS
        self.vlm_alternate_half_interval = VLM_ALTERNATE_HALF_INTERVAL

        if self.vlm_use_two_models:
            print("Initializing two VLM model instances for alternating inference...")
            self.vlm_classifier_1 = VLMActivityClassifier()
            self.vlm_classifier_2 = VLMActivityClassifier()
            self.vlm_model_index = 0
            # If alternating halves the classify interval, adjust effective interval
            eff_interval = VLM_CLASSIFY_INTERVAL / 2.0 if self.vlm_alternate_half_interval else VLM_CLASSIFY_INTERVAL
            self.vlm_classify_interval = eff_interval
        else:
            self.vlm_classifier = VLMActivityClassifier()
            self.vlm_classifier_1 = None
            self.vlm_classifier_2 = None
            self.vlm_model_index = 0
        # Ensure classify interval is set for single-model case
        if not hasattr(self, 'vlm_classify_interval'):
            self.vlm_classify_interval = VLM_CLASSIFY_INTERVAL
        self.activity_logger = ActivityLogger()
        self.activity_tracker = ActivityTracker()
        self.decision_engine = ActivityDecisionEngine()
        self.video_handler = VideoCapture(
            source=video_source,
            fps=OUTPUT_FPS,
            resize_scale=PROCESSING_DOWNSCALE
        )
        self.frame_renderer = FrameRenderer()
        
        # Initialize YOLO detector
        self.enable_yolo = enable_yolo
        self.yolo_detector = None
        if enable_yolo:
            try:
                # Find YOLO model
                model_path = "yolov8n.pt"
                if not os.path.exists(model_path):
                    model_path = os.path.join("..", model_path)
                
                self.yolo_detector = YOLOObjectDetector(model_path)
                print(f"YOLO detector initialized with {model_path}")
            except Exception as e:
                print(f"Warning: Could not initialize YOLO detector: {e}")
                self.enable_yolo = False
        
        # Frame sampling: process at ~TARGET_PROCESS_FPS
        input_fps = self.video_handler.get_input_fps() or TARGET_PROCESS_FPS
        self.frame_interval = max(1, int(round(input_fps / TARGET_PROCESS_FPS)))
        self.frame_index = 0
        
        # Timing
        self.prev_classify_time = time.time()
        self.activity_tracker.activity_start_time = time.time()
        
        # FPS Tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.fps_samples = []  # Track FPS for average calculation
        
        # VLM classification interval options (dynamic)
        # Note: when using two models with half-interval enabled, self.vlm_classify_interval
        # was already set above to VLM_CLASSIFY_INTERVAL/2.
        self.vlm_interval_index = 0  # Cycle through options
        self.vlm_interval_options = [0.5, 1.0, 2.0]

        # VLM inference counters for monitoring
        self.vlm_inference_count = 0
        self.vlm_inference_start_time = time.time()
        
        # Visualization control
        self.show_mediapipe = False  # Toggle MediaPipe landmarks visualization
        self.show_yolo = enable_yolo  # Toggle YOLO detections visualization
        self.show_touch_overlay = True  # Toggle hand-to-object overlay
        
        print("System initialized successfully!")
        print(f"Video dimensions: {self.video_handler.get_dimensions()}")
        print(f"Source FPS: {input_fps:.2f}, processing every {self.frame_interval} frame(s)")
        print(f"YOLO detection: {'ENABLED' if self.enable_yolo else 'DISABLED'}")
    
    def run(self):
        """Run the activity tracking system"""
        print("Starting activity tracking...")
        print("Controls:")
        print("  'q' = quit")
        print("  'm' = toggle MediaPipe landmarks")
        print("  'y' = toggle YOLO object detection")
        print("  'c' = cycle VLM interval (0.5s / 1.0s / 2.0s)")
        print("  'o' = toggle hand-object overlay")
        print("")
        
        try:
            while True:
                ret, frame = self.video_handler.read()
                if not ret:
                    print("End of video or camera disconnected")
                    break
                
                self.frame_index += 1
                if self.frame_index % self.frame_interval != 0:
                    continue
                
                current_time = time.time()
                
                # Update FPS counter
                self.fps_counter += 1
                elapsed_fps_time = current_time - self.fps_start_time
                if elapsed_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / elapsed_fps_time
                    self.fps_samples.append(self.current_fps)  # Add to samples for averaging
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    print(f"Processing FPS: {self.current_fps:.2f} | VLM Interval: {self.vlm_classify_interval:.1f}s")
                
                # Process frame with motion detection
                detection_info = self.motion_detector.process_frame(frame)
                
                # Run YOLO object detection
                yolo_detections = None
                if self.enable_yolo and self.yolo_detector:
                    yolo_detections = self.yolo_detector.detect_objects(frame)
                
                # Run VLM classification at dynamic interval
                if current_time - self.prev_classify_time >= self.vlm_classify_interval:
                    # Choose VLM model (alternate if two-model mode enabled)
                    try:
                        if self.vlm_use_two_models and self.vlm_classifier_1 and self.vlm_classifier_2:
                            classifier = self.vlm_classifier_1 if self.vlm_model_index == 0 else self.vlm_classifier_2
                            model_label = f"vlm{self.vlm_model_index+1}"
                            # Toggle for next time
                            self.vlm_model_index = 1 - self.vlm_model_index
                        else:
                            classifier = self.vlm_classifier
                            model_label = "vlm"

                        # Run inference
                        vlm_result = classifier.classify(frame)
                        vlm_result = classifier.post_process_vlm_result(vlm_result)

                        # Bookkeeping for VLM inference rate
                        self.vlm_inference_count += 1
                        elapsed_vlm_time = time.time() - self.vlm_inference_start_time
                        vlm_fps = (self.vlm_inference_count / elapsed_vlm_time) if elapsed_vlm_time > 0 else 0.0
                        # Print VLM inference rate and model used
                        print(f"VLM [{model_label}] result: {vlm_result} | VLM rate: {vlm_fps:.2f} inf/s")

                        # Log GPU usage after VLM inference for monitoring (label by model)
                        log_gpu_usage(label=f'{model_label}_inference')

                    except Exception as e:
                        print(f"Warning: VLM inference failed: {e}")
                        vlm_result = None
                    self.prev_classify_time = current_time
                    
                    # Include YOLO detections in detection_info for richer rules
                    detection_info['yolo_detections'] = yolo_detections

                    # Compute hand-landmark proximity to YOLO boxes (assembly indicator)
                    hand_touching_assembly = False
                    hand_touching_details = []
                    if yolo_detections and isinstance(yolo_detections, dict):
                        boxes = yolo_detections.get('object_boxes', [])
                        h_img, w_img = frame.shape[:2]
                        assembly_keywords = (
                            'drone', 'propell', 'screw', 'tool', 'motor', 'battery', 'part', 'assembly'
                        )
                        hand_results = detection_info.get('hand_results')
                        if hand_results and getattr(hand_results, 'multi_hand_landmarks', None):
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                for lm in hand_landmarks.landmark:
                                    x_px = int(lm.x * w_img)
                                    y_px = int(lm.y * h_img)
                                    for obj in boxes:
                                        cls = obj.get('class', '').lower()
                                        if any(k in cls for k in assembly_keywords):
                                            x1, y1, x2, y2 = obj['box']
                                            pad_x = int((x2 - x1) * 0.2)
                                            pad_y = int((y2 - y1) * 0.2)
                                            if (x1 - pad_x) <= x_px <= (x2 + pad_x) and (y1 - pad_y) <= y_px <= (y2 + pad_y):
                                                hand_touching_assembly = True
                                                hand_touching_details.append({'box': obj, 'hand_point': (x_px, y_px)})
                                                break
                                    if hand_touching_assembly:
                                        break
                                if hand_touching_assembly:
                                    break

                    detection_info['hand_touching_assembly'] = hand_touching_assembly
                    detection_info['hand_touching_details'] = hand_touching_details

                    # Ensure vlm_result fallback
                    if not vlm_result:
                        vlm_result = ACTIVITY_UNKNOWN

                    # Combine VLM and motion analysis (decision engine may use YOLO evidence)
                    new_activity = self.decision_engine.classify_activity(vlm_result, detection_info)
                    
                    # Update activity tracker
                    activity_changed, old_activity, _ = self.activity_tracker.update_activity(
                        new_activity, current_time, PHONE_RESET_FRAMES
                    )
                    
                    # Log activity transition with average FPS
                    if activity_changed:
                        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
                        self.activity_logger.log_activity(
                            old_activity,
                            self.activity_tracker.activity_start_time,
                            current_time,
                            avg_fps=avg_fps
                        )
                        self.activity_tracker.activity_start_time = current_time
                        self.fps_samples = []  # Reset FPS samples for new activity
                    
                    # Debug output
                    self.decision_engine.print_motion_debug_info(
                        detection_info.get('motion_stats'), detection_info
                    )
                
                # Get elapsed time for current activity
                elapsed_time = self.activity_tracker.get_elapsed_time(current_time)
                
                # Generate alert message if needed
                alert_message = self.decision_engine.get_alert_message(
                    self.activity_tracker.current_activity, elapsed_time
                )
                
                # Draw MediaPipe landmarks if enabled
                if self.show_mediapipe:
                    frame = self.motion_detector.draw_landmarks(
                        frame,
                        detection_info['hand_results'],
                        detection_info['pose_results']
                    )
                
                # Draw YOLO detections if enabled
                if self.show_yolo and yolo_detections:
                    frame = self.yolo_detector.draw_detections(frame, yolo_detections)

                # Draw hand-to-object overlay for tuning if hand_touching_details present
                touch_details = detection_info.get('hand_touching_details', [])
                if touch_details and self.show_touch_overlay:
                    for td in touch_details:
                        obj = td.get('box')
                        hand_pt = td.get('hand_point')
                        try:
                            if obj and hand_pt:
                                x1, y1, x2, y2 = obj['box']
                                # Draw highlighted box (orange) and hand point (red)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
                                cv2.circle(frame, hand_pt, 6, (0, 0, 255), -1)
                                # Draw line from hand point to center of box
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                cv2.line(frame, hand_pt, (cx, cy), (0, 128, 255), 1)
                                # Label
                                label = f"{obj.get('class','object')} touch"
                                cv2.putText(frame, label, (x1, max(12, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                        except Exception:
                            # Defensive: skip any malformed entries
                            continue
                
                # Render information on frame
                frame = self.frame_renderer.render_activity_info(
                    frame,
                    self.activity_tracker.current_activity,
                    elapsed_time,
                    alert_message,
                    fps=self.current_fps,
                    show_mediapipe=self.show_mediapipe,
                    show_yolo=self.show_yolo
                )
                
                # Display and write frame
                cv2.imshow("Drone Assembly Monitoring", frame)
                self.video_handler.write(frame)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord('m'):
                    self.show_mediapipe = not self.show_mediapipe
                    status = "ON" if self.show_mediapipe else "OFF"
                    print(f"MediaPipe visualization: {status}")
                elif key == ord('y'):
                    if self.enable_yolo:
                        self.show_yolo = not self.show_yolo
                        status = "ON" if self.show_yolo else "OFF"
                        print(f"YOLO detection visualization: {status}")
                elif key == ord('o'):
                    self.show_touch_overlay = not self.show_touch_overlay
                    status = "ON" if self.show_touch_overlay else "OFF"
                    print(f"Hand-object overlay: {status}")
                elif key == ord('c'):
                    self.vlm_interval_index = (self.vlm_interval_index + 1) % len(self.vlm_interval_options)
                    base = self.vlm_interval_options[self.vlm_interval_index]
                    if self.vlm_use_two_models and self.vlm_alternate_half_interval:
                        self.vlm_classify_interval = base / 2.0
                    else:
                        self.vlm_classify_interval = base
                    print(f"VLM classification interval changed to {self.vlm_classify_interval:.1f}s (base {base:.1f}s)")
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the system and cleanup resources"""
        print("Shutting down system...")
        
        # Log final activity with average FPS
        if self.activity_tracker.activity_start_time:
            avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0
            self.activity_logger.log_activity(
                self.activity_tracker.current_activity,
                self.activity_tracker.activity_start_time,
                time.time(),
                avg_fps=avg_fps
            )

        # Log GPU usage at shutdown
        log_gpu_usage(label='shutdown')
        
        # Cleanup resources
        self.motion_detector.release()
        # Release VLM models
        if self.vlm_classifier_1:
            self.vlm_classifier_1.release()
        if self.vlm_classifier_2:
            self.vlm_classifier_2.release()
        if hasattr(self, 'vlm_classifier') and self.vlm_classifier:
            self.vlm_classifier.release()
        if self.yolo_detector:
            self.yolo_detector.release()
        self.video_handler.release()
        
        # Print summary
        summary = self.activity_logger.get_session_summary()
        if summary is not None:
            print("\nSession Summary:")
            print(summary)
        
        print("System shutdown complete")


def main():
    """Main entry point"""
    # For webcam use: ActivityTrackingSystem(video_source=0)
    # For video file use: ActivityTrackingSystem(video_source="path/to/video.mp4")
    
    # system = ActivityTrackingSystem(0);
    system = ActivityTrackingSystem(
        video_source="C:\\Users\\BBBS-AI-01\\d\\behavior_tracking\\data\\assembly_idle.mp4",
        enable_yolo=True
    )
    system.run()


if __name__ == "__main__":
    main()

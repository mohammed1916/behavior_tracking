"""Main activity tracking system orchestrator"""

import time
import cv2
from motion_detector import MediaPipeMotionDetector
from vlm_classifier import VLMActivityClassifier
from activity_logger import ActivityLogger, ActivityTracker
from decision_engine import ActivityDecisionEngine
from video_handler import VideoCapture, FrameRenderer
from config import VLM_CLASSIFY_INTERVAL, PHONE_RESET_FRAMES, TARGET_PROCESS_FPS, PROCESSING_DOWNSCALE, OUTPUT_FPS


class ActivityTrackingSystem:
    """Main system that orchestrates activity tracking"""
    
    def __init__(self, video_source=0):
        """Initialize the activity tracking system
        
        Args:
            video_source: Video source (0 for webcam, or file path)
        """
        print("Initializing Activity Tracking System...")
        
        # Initialize components
        self.motion_detector = MediaPipeMotionDetector()
        self.vlm_classifier = VLMActivityClassifier()
        self.activity_logger = ActivityLogger()
        self.activity_tracker = ActivityTracker()
        self.decision_engine = ActivityDecisionEngine()
        self.video_handler = VideoCapture(
            source=video_source,
            fps=OUTPUT_FPS,
            resize_scale=PROCESSING_DOWNSCALE
        )
        self.frame_renderer = FrameRenderer()
        
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
        
        print("System initialized successfully!")
        print(f"Video dimensions: {self.video_handler.get_dimensions()}")
        print(f"Source FPS: {input_fps:.2f}, processing every {self.frame_interval} frame(s)")
    
    def run(self):
        """Run the activity tracking system"""
        print("Starting activity tracking... Press 'q' to quit.")
        
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
                    print(f"Processing FPS: {self.current_fps:.2f}")
                
                # Process frame with motion detection
                detection_info = self.motion_detector.process_frame(frame)
                
                # Run VLM classification once per second
                if current_time - self.prev_classify_time >= VLM_CLASSIFY_INTERVAL:
                    vlm_result = self.vlm_classifier.classify(frame)
                    vlm_result = self.vlm_classifier.post_process_vlm_result(vlm_result)
                    self.prev_classify_time = current_time
                    
                    # Combine VLM and motion analysis
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
                
                # Render information on frame
                frame = self.frame_renderer.render_activity_info(
                    frame,
                    self.activity_tracker.current_activity,
                    elapsed_time,
                    alert_message,
                    fps=self.current_fps
                )
                
                # Optional: Draw motion landmarks (uncomment to visualize)
                # frame = self.motion_detector.draw_landmarks(
                #     frame,
                #     detection_info['hand_results'],
                #     detection_info['pose_results']
                # )
                
                # Display and write frame
                cv2.imshow("Drone Assembly Monitoring", frame)
                self.video_handler.write(frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
        
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
        
        # Cleanup resources
        self.motion_detector.release()
        self.vlm_classifier.release()
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
    
    system = ActivityTrackingSystem(video_source="C:\\Users\\BBBS-AI-01\\d\\behavior_tracking\\data\\assembly_idle.mp4")
    system.run()


if __name__ == "__main__":
    main()

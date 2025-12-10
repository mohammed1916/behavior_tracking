"""Video capture and processing utilities"""

import cv2
from config import VIDEO_FPS, VIDEO_CODEC, VIDEO_OUTPUT_FILE, OUTPUT_FPS


class VideoCapture:
    """Wrapper for video capture with optional downscaling and output recording"""
    
    def __init__(self, source=0, output_file=VIDEO_OUTPUT_FILE, fps=OUTPUT_FPS, codec=VIDEO_CODEC, resize_scale=1.0):
        """Initialize video capture
        
        Args:
            source: Video source (0 for webcam, or file path)
            output_file: Path for output video file
            fps: Frames per second for output
            codec: Video codec (e.g., 'mp4v', 'XVID')
            resize_scale: Scale factor to downsize frames for processing/output
        """
        self.cap = cv2.VideoCapture(source)
        self.resize_scale = resize_scale
        
        # Get source video properties
        self.input_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.input_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        
        # Apply scaling for processing/output
        self.width = max(1, int(self.input_width * self.resize_scale)) if self.input_width else int(640 * self.resize_scale)
        self.height = max(1, int(self.input_height * self.resize_scale)) if self.input_height else int(480 * self.resize_scale)
        self.fps = fps
        
        # Initialize video writer sized to processed frames
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(
            output_file,
            fourcc,
            fps,
            (self.width, self.height)
        )
    
    def read(self):
        """Read next frame from video and optionally downscale it
        
        Returns:
            tuple: (success, frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            return ret, frame
        if self.resize_scale != 1.0:
            frame = cv2.resize(frame, (self.width, self.height))
        return ret, frame
    
    def write(self, frame):
        """Write frame to output video
        
        Args:
            frame: Frame to write
        """
        self.out.write(frame)
    
    def release(self):
        """Release video capture and writer"""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
    
    def get_dimensions(self):
        """Get video dimensions
        
        Returns:
            tuple: (width, height)
        """
        return self.width, self.height

    def get_input_fps(self):
        """Get source video FPS (falls back to configured fps if unknown)"""
        return self.input_fps


class FrameRenderer:
    """Handles rendering of activity information on frames"""
    
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX):
        """Initialize frame renderer
        
        Args:
            font: OpenCV font type
        """
        self.font = font
        self.font_scale = 1
        self.thickness = 2
        self.thickness_alert = 4
        self.color_info = (0, 255, 0)      # Green
        self.color_timer = (255, 255, 0)   # Yellow
        self.color_alert = (0, 255, 0)     # Green
    
    def render_activity_info(self, frame, activity, elapsed_time, alert_message="", fps=0, show_mediapipe=False, show_yolo=False):
        """Render activity information on frame
        
        Args:
            frame: Frame to render on
            activity: Current activity label
            elapsed_time: Elapsed time for activity (seconds)
            alert_message: Alert message to display (optional)
            fps: Current processing FPS (optional)
            show_mediapipe: Whether MediaPipe visualization is enabled (optional)
            show_yolo: Whether YOLO visualization is enabled (optional)
        
        Returns:
            frame: Rendered frame
        """
        # Activity label
        cv2.putText(
            frame,
            f"Activity: {activity}",
            (30, 40),
            self.font,
            self.font_scale,
            self.color_info,
            self.thickness
        )
        
        # Timer
        cv2.putText(
            frame,
            f"Time: {int(elapsed_time)} sec",
            (30, 80),
            self.font,
            self.font_scale,
            self.color_timer,
            self.thickness
        )
        
        # FPS display
        if fps > 0:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (30, 120),
                self.font,
                self.font_scale,
                self.color_info,
                self.thickness
            )
        
        # MediaPipe status indicator
        y_offset = 160 if fps > 0 else 120
        mediapipe_status = "MediaPipe: ON" if show_mediapipe else "MediaPipe: OFF"
        cv2.putText(
            frame,
            mediapipe_status,
            (30, y_offset),
            self.font,
            0.7,  # Smaller font
            (255, 255, 255) if show_mediapipe else (128, 128, 128),  # White if on, gray if off
            self.thickness
        )
        
        # YOLO status indicator
        yolo_status = "YOLO: ON" if show_yolo else "YOLO: OFF"
        cv2.putText(
            frame,
            yolo_status,
            (30, y_offset + 25),
            self.font,
            0.7,  # Smaller font
            (0, 255, 255) if show_yolo else (128, 128, 128),  # Yellow if on, gray if off
            self.thickness
        )
        
        # Alert message
        if alert_message:
            alert_y = y_offset + 50
            cv2.putText(
                frame,
                alert_message,
                (30, alert_y),
                self.font,
                self.font_scale,
                self.color_alert,
                self.thickness_alert
            )
        
        return frame
    
    def render_motion_visualization(self, frame, detection_info):
        """Render motion and pose visualization on frame
        
        Args:
            frame: Frame to render on
            detection_info: Detection information with hand/pose results
        
        Returns:
            frame: Rendered frame with landmarks
        """
        # This would use the draw_landmarks method from motion_detector
        # Can be extended to show motion regions, hand positions, etc.
        return frame

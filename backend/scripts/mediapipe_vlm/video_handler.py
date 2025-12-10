"""Video capture and processing utilities"""

import cv2
from config import VIDEO_FPS, VIDEO_CODEC, VIDEO_OUTPUT_FILE


class VideoCapture:
    """Wrapper for video capture with output recording"""
    
    def __init__(self, source=0, output_file=VIDEO_OUTPUT_FILE, fps=VIDEO_FPS, codec=VIDEO_CODEC):
        """Initialize video capture
        
        Args:
            source: Video source (0 for webcam, or file path)
            output_file: Path for output video file
            fps: Frames per second for output
            codec: Video codec (e.g., 'mp4v', 'XVID')
        """
        self.cap = cv2.VideoCapture(source)
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(
            output_file,
            fourcc,
            fps,
            (self.width, self.height)
        )
    
    def read(self):
        """Read next frame from video
        
        Returns:
            tuple: (success, frame)
        """
        return self.cap.read()
    
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
    
    def render_activity_info(self, frame, activity, elapsed_time, alert_message=""):
        """Render activity information on frame
        
        Args:
            frame: Frame to render on
            activity: Current activity label
            elapsed_time: Elapsed time for activity (seconds)
            alert_message: Alert message to display (optional)
        
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
        
        # Alert message
        if alert_message:
            cv2.putText(
                frame,
                alert_message,
                (30, 120),
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

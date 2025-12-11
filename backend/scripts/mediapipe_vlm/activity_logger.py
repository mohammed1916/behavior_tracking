"""Activity logging and tracking functionality"""

import pandas as pd
from datetime import datetime
from config import ACTIVITY_CSV_FILE


class ActivityLogger:
    """Handles logging of activities to CSV"""
    
    def __init__(self, csv_file=ACTIVITY_CSV_FILE):
        """Initialize activity logger
        
        Args:
            csv_file: Path to CSV file for logging
        """
        self.csv_file = csv_file
        self.log_data = []
    
    def log_activity(self, activity, start_time, end_time, avg_fps=0):
        """Log an activity session
        
        Args:
            activity: Activity label (string)
            start_time: Start timestamp (seconds since epoch)
            end_time: End timestamp (seconds since epoch)
            avg_fps: Average FPS during activity (optional)
        """
        duration = round(end_time - start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.log_data.append([timestamp, activity, duration, round(avg_fps, 2)])
        
        # Write to CSV
        df = pd.DataFrame(self.log_data, columns=["timestamp", "activity", "duration_sec", "avg_fps"])
        df.to_csv(self.csv_file, index=False)
        
        print(f"Logged: {activity} for {duration}s (avg FPS: {avg_fps:.2f}) at {timestamp}")
    
    def get_session_summary(self):
        """Get summary of logged activities"""
        if not self.log_data:
            return None
        
        df = pd.DataFrame(self.log_data, columns=["timestamp", "activity", "duration_sec", "avg_fps"])
        return df


class ActivityTracker:
    """Tracks current activity state"""
    
    def __init__(self):
        """Initialize activity tracker"""
        self.current_activity = "unknown"
        self.activity_start_time = None
        self.phone_missing_frames = 0
    
    def update_activity(self, new_activity, current_time, phone_reset_frames):
        """Update activity state and return if activity changed
        
        Args:
            new_activity: New activity label
            current_time: Current timestamp (seconds)
            phone_reset_frames: Number of frames to wait before resetting phone detection
        
        Returns:
            tuple: (activity_changed, old_activity, new_activity)
        """
        activity_changed = False
        old_activity = self.current_activity
        
        # Phone timer reset logic
        if self.current_activity == "using_phone":
            if new_activity != "using_phone":
                self.phone_missing_frames += 1
            else:
                self.phone_missing_frames = 0
            
            # Reset only if phone is missing for enough frames
            if self.phone_missing_frames >= phone_reset_frames:
                activity_changed = True
                self.current_activity = new_activity
                self.activity_start_time = current_time
                self.phone_missing_frames = 0
        else:
            # Normal activity change
            if new_activity != self.current_activity:
                activity_changed = True
                self.current_activity = new_activity
                self.activity_start_time = current_time
                self.phone_missing_frames = 0
        
        return activity_changed, old_activity, new_activity
    
    def get_elapsed_time(self, current_time):
        """Get elapsed time for current activity
        
        Args:
            current_time: Current timestamp (seconds)
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.activity_start_time is None:
            return 0.0
        return current_time - self.activity_start_time

"""Activity classification decision logic combining VLM and motion analysis"""

from config import (
    ACTIVITY_ASSEMBLING_DRONE, ACTIVITY_IDLE, ACTIVITY_USING_PHONE,
    ACTIVITY_SIMPLY_SITTING, ACTIVITY_UNKNOWN,
    IDLE_LIMIT, PHONE_LIMIT, DRONE_LIMIT
)


class ActivityDecisionEngine:
    """Combines VLM predictions with motion analysis for robust classification"""
    
    def __init__(self):
        """Initialize decision engine"""
        pass
    
    def classify_activity(self, vlm_result, detection_info):
        """Make final activity classification decision
        
        Args:
            vlm_result: Result from VLM classifier (post-processed)
            detection_info: Output from motion detector (contains motion and pose info)
        
        Returns:
            str: Final activity classification
        """
        motion_stats = detection_info.get('motion_stats')
        is_productive_motion = detection_info.get('is_productive_motion', False)
        pose_info = detection_info.get('pose_info')
        hand_motion_regions = detection_info.get('hand_motion_regions', 0)
        
        # Check phone usage from pose
        if pose_info and pose_info.get('wrists_near_face'):
            return ACTIVITY_USING_PHONE
        
        # Check VLM result for phone
        if "phone" in vlm_result:
            return ACTIVITY_USING_PHONE
        
        # Check for drone assembly from VLM
        if "assemble" in vlm_result or "drone" in vlm_result:
            return ACTIVITY_ASSEMBLING_DRONE
        
        # Enhanced decision making with motion analysis
        if motion_stats:
            # Strong productive motion pattern detected
            if is_productive_motion:
                # Override VLM with motion evidence if it's productive work
                if vlm_result in [ACTIVITY_IDLE, ACTIVITY_UNKNOWN]:
                    return ACTIVITY_ASSEMBLING_DRONE
                return ACTIVITY_ASSEMBLING_DRONE
            
            # Minimal motion detected
            if motion_stats['overall'] < 2.0:
                return ACTIVITY_IDLE
            
            # Medium motion but not productive pattern
            if 2.0 <= motion_stats['overall'] < 8.0:
                if vlm_result == ACTIVITY_IDLE:
                    return ACTIVITY_SIMPLY_SITTING
                elif vlm_result == ACTIVITY_UNKNOWN:
                    return ACTIVITY_SIMPLY_SITTING
            
            # High overall motion but not in work area (body movement)
            if motion_stats['overall'] > 8.0 and motion_stats['work_ratio'] < 1.0:
                return ACTIVITY_SIMPLY_SITTING
        
        # Fallback to VLM result
        if vlm_result == ACTIVITY_IDLE:
            return ACTIVITY_IDLE
        elif vlm_result == ACTIVITY_UNKNOWN:
            return ACTIVITY_SIMPLY_SITTING
        else:
            return vlm_result
    
    def get_alert_message(self, current_activity, elapsed_time):
        """Generate alert message based on activity and duration
        
        Args:
            current_activity: Current activity label
            elapsed_time: Elapsed time for current activity (seconds)
        
        Returns:
            str: Alert message (empty if no alert)
        """
        alert_message = ""
        
        if current_activity == "working" and elapsed_time > IDLE_LIMIT:
            alert_message = "Please assemble drone"
        
        elif current_activity == ACTIVITY_SIMPLY_SITTING and elapsed_time > IDLE_LIMIT:
            alert_message = f"Simply sitting for {int(elapsed_time)} sec"
        
        elif current_activity == ACTIVITY_IDLE and elapsed_time > IDLE_LIMIT:
            alert_message = f"Idle for {int(elapsed_time)} sec"
        
        elif current_activity == ACTIVITY_USING_PHONE and elapsed_time > PHONE_LIMIT:
            alert_message = f"Phone usage for {int(elapsed_time)} sec"
        
        elif current_activity == ACTIVITY_ASSEMBLING_DRONE and elapsed_time > DRONE_LIMIT:
            alert_message = "Drone usage limit exceeded"
        
        return alert_message
    
    def print_motion_debug_info(self, motion_stats, detection_info):
        """Print debug information for motion analysis
        
        Args:
            motion_stats: Motion statistics dictionary
            detection_info: Detection information dictionary
        """
        if motion_stats:
            print(f"Motion - Overall: {motion_stats['overall']:.2f}, "
                  f"Work Area: {motion_stats['work_area']:.2f}, "
                  f"Hand Score: {motion_stats['hand_score']}, "
                  f"Productive: {detection_info.get('is_productive_motion', False)}")

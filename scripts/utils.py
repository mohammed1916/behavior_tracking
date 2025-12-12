import cv2
import math
import numpy as np

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_point_in_box(point, box):
    """
    Checks if a point (x, y) is inside a bounding box [x1, y1, x2, y2].
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_info_panel(image, state, time_left, task_completed):
    """
    Draws the status panel on the image.
    """
    h, w, _ = image.shape
    
    # Panel background
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # State text
    color = (0, 255, 0) # Green for IDLE
    if state == "HOLDING":
        color = (0, 255, 255) # Yellow
    elif state == "OPENING":
        color = (0, 0, 255) # Red
    elif state == "COMPLETED":
        color = (255, 0, 0) # Blue
        
    cv2.putText(image, f"State: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Timer
    if time_left is not None:
        cv2.putText(image, f"Time: {time_left:.1f}s", (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    # Task Status
    status_text = "Task: Pending"
    if task_completed:
        status_text = "Task: DONE"
        cv2.putText(image, status_text, (w - 500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, status_text, (w - 500, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

def boxes_intersect(box1, box2):
    """
    Check if two boxes intersect.
    Box format: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    if (x1_min < x2_max and x1_max > x2_min and
        y1_min < y2_max and y1_max > y2_min):
        return True
    return False

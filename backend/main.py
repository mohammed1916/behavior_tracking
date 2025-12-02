import cv2
from tracker import BehaviorTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = BehaviorTracker()
    
    print("Starting video loop. Press 'q' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Process
        output_frame, state, completed = tracker.process_frame(frame)
        
        cv2.imshow('Behavior Tracking', output_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

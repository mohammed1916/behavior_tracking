# MediaPipe VLM Activity Tracking System

A modular activity tracking system that combines MediaPipe pose/hand detection with Qwen2-VL vision language model for accurate activity recognition.

## Architecture

The system is split into the following modules:

### Core Modules

#### `config.py`
Central configuration file containing all constants:
- Model names and device settings
- Motion detection thresholds
- Activity tracking limits
- Video output settings
- Activity labels

#### `motion_detector.py`
MediaPipe-based motion and pose detection using:
- **Hand Detection**: Tracks hand positions and velocity for work activity
- **Pose Detection**: Detects body landmarks (wrists, ears) for phone usage patterns
- **Optical Flow**: Analyzes pixel-level motion in different frame regions
- **Motion Analysis**: Classifies motion as productive or non-productive

Key classes:
- `MediaPipeMotionDetector`: Main detector with frame processing and landmark drawing

#### `vlm_classifier.py`
Vision Language Model classifier using Qwen2-VL 2B:
- Image-to-text activity classification
- Post-processing of VLM outputs to standard labels
- GPU acceleration support

Key classes:
- `VLMActivityClassifier`: Handles model loading and inference

#### `decision_engine.py`
Decision logic that combines multiple signals:
- Fuses VLM predictions with motion analysis
- Detects phone usage from pose landmarks
- Generates alert messages for time limits
- Provides override logic (e.g., motion evidence overrides VLM when confident)

Key classes:
- `ActivityDecisionEngine`: Makes final activity classification decisions

#### `activity_logger.py`
Activity tracking and CSV logging:
- Tracks activity state and transitions
- Logs activities to CSV with timestamps and durations
- Manages phone detection debouncing

Key classes:
- `ActivityLogger`: CSV logging functionality
- `ActivityTracker`: State management for current activity

#### `video_handler.py`
Video capture and frame rendering:
- Webcam or video file capture
- Video output recording
- Frame rendering with activity information and alerts

Key classes:
- `VideoCapture`: Handles video I/O
- `FrameRenderer`: Renders activity info and alerts on frames

#### `main.py`
Main orchestrator that ties everything together:
- Initializes all components
- Runs the main processing loop
- Handles frame processing pipeline
- Manages cleanup and shutdown

Key classes:
- `ActivityTrackingSystem`: Main system orchestrator

## Activity Recognition Flow

```
Frame Input
    ↓
Motion Detection (MediaPipe + Optical Flow)
    ↓
[Every 1 second] VLM Classification
    ↓
Decision Engine (Fusion Logic)
    ↓
Activity Tracking & Logging
    ↓
Frame Rendering & Output
```

## Activity Labels

- **assembling_drone**: Working with tools, handling parts, tightening screws
- **idle**: Standing/sitting without task, arms resting
- **using_phone**: Holding or interacting with phone
- **simply_sitting**: Some movement but not productive work
- **unknown**: Activity cannot be confidently identified

## Alert Conditions

- **Idle > 10 seconds**: "Idle for X sec"
- **Phone usage > 10 seconds**: "Phone usage for X sec"
- **Assembling drone > 20 seconds**: "Drone usage limit exceeded"
- **Simply sitting > 10 seconds**: "Simply sitting for X sec"

## Configuration

Edit `config.py` to adjust:
- Motion detection thresholds (`WORK_MOTION_MIN`, `WORK_MOTION_MAX`, etc.)
- Activity time limits (`IDLE_LIMIT`, `PHONE_LIMIT`, `DRONE_LIMIT`)
- MediaPipe confidence levels
- Output file names and formats

## Usage

### Running with Webcam
```python
from main import ActivityTrackingSystem

system = ActivityTrackingSystem(video_source=0)
system.run()
```

### Running with Video File
```python
system = ActivityTrackingSystem(video_source="path/to/video.mp4")
system.run()
```

### Running from Command Line
```bash
python main.py
```

## Output

- **activity_log.csv**: Log of all activities with timestamps and durations
- **data_collection_.mp4**: Video output with rendered activity information

## Requirements

- PyTorch
- transformers (Hugging Face)
- MediaPipe
- OpenCV
- pandas
- PIL
- numpy

Install with:
```bash
pip install torch transformers mediapipe opencv-python pandas pillow numpy
```

## Key Features

✅ **MediaPipe Integration**: Hand and pose tracking for robust motion detection
✅ **VLM Classification**: State-of-the-art vision language model for activity recognition
✅ **Fusion Architecture**: Combines multiple signals for improved accuracy
✅ **Modular Design**: Easy to extend and maintain
✅ **Real-time Processing**: GPU-accelerated inference
✅ **Activity Logging**: CSV export of activity sessions
✅ **Alert System**: Time-based alerts for productivity monitoring

## Extending the System

### Add a New Activity
1. Add activity label to `config.py`
2. Update VLM prompt in `vlm_classifier.py`
3. Add decision logic in `decision_engine.py`

### Add Custom Motion Detection
Extend `MediaPipeMotionDetector` in `motion_detector.py`:
```python
def detect_custom_motion(self, frame, hand_results):
    # Your custom motion detection logic
    pass
```

### Add Custom Alerts
Extend `get_alert_message()` in `decision_engine.py`

## Performance Notes

- VLM classification runs once per second to balance accuracy and speed
- Motion detection runs every frame for real-time feedback
- Requires GPU for optimal performance (falls back to CPU if unavailable)
- ~1-2 GB VRAM required for Qwen2-VL 2B

## Troubleshooting

**Model Download Issues**: Models are auto-downloaded from Hugging Face on first run
**Camera Not Found**: Verify camera connection and OpenCV support
**Out of Memory**: Reduce frame resolution or run on GPU with more VRAM
**Slow Classification**: Check GPU availability with `torch.cuda.is_available()`

## Original Code

The original monolithic `vlm_rules.py` has been refactored into this modular structure while maintaining all functionality.

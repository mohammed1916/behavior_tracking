# ML Task Classification Training Pipeline

This document describes the complete pipeline for training ML models to identify different tasks from MediaPipe + YOLO detector outputs.

## Overview

The training pipeline has 4 main stages:

1. **Feature Extraction**: Extract per-frame features from videos using MediaPipe + YOLO
2. **Dataset Building**: Create sliding windows, normalize, balance classes, split train/val/test
3. **Model Training**: Train RandomForest or LightGBM baseline classifier
4. **Inference Integration**: Use trained model in streaming pipeline alongside VLM

## Pipeline Stages

### 1. Feature Extraction

**Script**: `scripts/extract_features.py`
**Endpoint**: `POST /backend/upload_and_extract_features`

Extracts per-frame features from video:

- **MediaPipe**: Pose keypoints (33x3), hand landmarks (2x21x3)
- **YOLO**: Object detections (class, bbox, confidence)
- **Derived**: Joint angles, hand velocities, hand-object distances

**Usage (CLI)**:

```bash
python scripts/extract_features.py \
  --video path/to/video.mp4 \
  --output processed/features/video_features.parquet \
  --labels labels.csv \
  --detector fusion \
  --sample-rate 1
```

**Usage (API)**:

```bash
curl -X POST http://localhost:8001/backend/upload_and_extract_features \
  -F "video=@video.mp4" \
  -F "labels_csv=@labels.csv" \
  -F "detector_type=fusion" \
  -F "sample_rate=1"
```

**Output**: Parquet file with columns:

- `video_id`, `frame_index`, `t` (timestamp)
- `mp_pose_landmarks`, `mp_left_hand_landmarks`, `mp_right_hand_landmarks` (JSON)
- `yolo_objects`, `yolo_boxes`, `yolo_confidences` (JSON)
- `right_elbow_angle`, `left_elbow_angle`, `right_hand_x`, `right_hand_y`, etc.
- `right_hand_speed`, `left_hand_speed`, `right_hand_obj_dist`, etc.
- `label` (if annotations provided)

---

### 2. Dataset Building

**Script**: `scripts/build_task_dataset.py`
**Endpoint**: `POST /backend/build_task_dataset`

Creates train/val/test splits with sliding windows:

- Sliding windows (default: 2s window, 0.5s step)
- Z-score normalization per feature
- Class balancing (downsample or upsample)
- Video-level splits (prevent leakage)

**Usage (CLI)**:

```bash
python scripts/build_task_dataset.py \
  --features processed/features/vid1.parquet processed/features/vid2.parquet \
  --output processed/datasets/dataset_001 \
  --window-sec 2.0 \
  --step-sec 0.5 \
  --balance downsample \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

**Usage (API)**:

```bash
curl -X POST http://localhost:8001/backend/build_task_dataset \
  -F "feature_files=vid1_features.parquet,vid2_features.parquet" \
  -F "window_sec=2.0" \
  -F "step_sec=0.5" \
  -F "balance_method=downsample"
```

**Output**: Directory with:

- `X_train.npy`, `y_train.npy` (training windows & labels)
- `X_val.npy`, `y_val.npy` (validation)
- `X_test.npy`, `y_test.npy` (test)
- `metadata.json` (feature names, scaler params, label mapping, splits info)

---

### 3. Model Training

**Script**: `scripts/train_task_classifier.py`
**Endpoint**: `POST /backend/train_task_model`

Trains baseline classifier:

- **RandomForest** or **LightGBM**
- Aggregation: `stats` (mean/std/min/max) or `flatten` (entire window)
- Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`

**Usage (CLI)**:

```bash
python scripts/train_task_classifier.py \
  --dataset processed/datasets/dataset_001 \
  --model rf \
  --output mlruns/models/task_classifier.pkl \
  --agg stats \
  --n-estimators 100
```

**Usage (API)**:

```bash
curl -X POST http://localhost:8001/backend/train_task_model \
  -F "dataset_dir=dataset_001" \
  -F "model_type=rf" \
  -F "aggregation=stats" \
  -F "n_estimators=100"
```

**Output**:

- `task_classifier_rf_abc123.pkl` (trained model + metadata)
- `task_classifier_rf_abc123_metrics.json` (accuracy, F1, confusion matrix, classification report)

**Metrics**:

- Accuracy, F1-macro, F1-weighted
- Per-class precision/recall/F1
- Confusion matrix
- Feature importance (for RandomForest)

---

### 4. Inference Integration

**Status**: Not yet implemented
**Planned**: Integrate trained model into `stream_processor.py` to emit task predictions alongside VLM captions

**Design**:

- Load model: `pickle.load('model.pkl')`
- Maintain rolling buffer of features (2s window)
- Every frame: aggregate window → predict task → emit alongside caption
- Smoothing: median filter or EMA to reduce prediction flicker

**API addition**:

```python
@app.get("/backend/stream_pose")
async def stream_pose(
    ...,
    task_model: str = Query(None),  # model filename from mlruns/models/
):
    # Load model if provided
    if task_model:
        model_data = load_task_model(task_model)
  
    # In stream loop:
    # - Maintain rolling buffer of features
    # - Aggregate window
    # - Predict task
    # - Emit: {"stage": "sample", "caption": "...", "task_prediction": "work", "task_confidence": 0.85}
```

---

## Label Format

### CSV Format (for feature extraction)

```csv
frame_index,label
0,idle
1,idle
...
150,work
151,work
...
```

### Supported Labels

- `idle`: No activity, resting
- `work`: Active task execution
- Custom: Any string labels (multi-class supported)

---

## API Endpoints Summary

| Endpoint                                 | Method | Purpose                                          |
| ---------------------------------------- | ------ | ------------------------------------------------ |
| `/backend/upload_and_extract_features` | POST   | Upload video → extract features → save Parquet |
| `/backend/features`                    | GET    | List all extracted feature files                 |
| `/backend/features/{filename}`         | GET    | Get metadata for feature file                    |
| `/backend/build_task_dataset`          | POST   | Build windowed dataset from features             |
| `/backend/datasets`                    | GET    | List all built datasets                          |
| `/backend/train_task_model`            | POST   | Train classifier on dataset                      |
| `/backend/models`                      | GET    | List all trained models                          |

---

## Example End-to-End Workflow

### Via API

```bash
# 1. Upload & extract features
curl -X POST http://localhost:8001/backend/upload_and_extract_features \
  -F "video=@assembly_video.mp4" \
  -F "labels_csv=@assembly_labels.csv" \
  -F "detector_type=fusion" \
  > extract_result.json

FEATURE_FILE=$(jq -r '.feature_file | split("/")[-1]' extract_result.json)

# 2. Build dataset
curl -X POST http://localhost:8001/backend/build_task_dataset \
  -F "feature_files=$FEATURE_FILE" \
  -F "window_sec=2.0" \
  -F "step_sec=0.5" \
  > dataset_result.json

DATASET_ID=$(jq -r '.dataset_id' dataset_result.json)

# 3. Train model
curl -X POST http://localhost:8001/backend/train_task_model \
  -F "dataset_dir=dataset_$DATASET_ID" \
  -F "model_type=rf" \
  -F "aggregation=stats" \
  -F "n_estimators=100" \
  > model_result.json

MODEL_ID=$(jq -r '.model_id' model_result.json)
echo "Trained model: task_classifier_rf_$MODEL_ID.pkl"
```

### Via CLI

```bash
# 1. Extract features
python scripts/extract_features.py \
  --video assembly_video.mp4 \
  --labels assembly_labels.csv \
  --output processed/features/assembly_features.parquet

# 2. Build dataset
python scripts/build_task_dataset.py \
  --features processed/features/assembly_features.parquet \
  --output processed/datasets/assembly_dataset

# 3. Train model
python scripts/train_task_classifier.py \
  --dataset processed/datasets/assembly_dataset \
  --model rf \
  --output mlruns/models/assembly_classifier.pkl
```

---

## Directory Structure

```
behavior_tracking/
├── scripts/
│   ├── extract_features.py        # Feature extraction
│   ├── build_task_dataset.py      # Dataset building
│   └── train_task_classifier.py   # Model training
├── backend/
│   ├── server.py                  # API endpoints
│   └── detectors/
│       ├── mediapipe_yolo_detector.py  # Fusion detector
│       ├── mediapipe_detector.py
│       └── yolo_detector.py
├── processed/
│   ├── features/                  # Extracted feature files (.parquet)
│   └── datasets/                  # Built datasets (X_train.npy, etc.)
├── mlruns/
│   └── models/                    # Trained models (.pkl)
└── vlm_uploads/                   # Uploaded videos
```

---

## Next Steps

- [x] Implement inference integration in `stream_processor.py`  
  **Done:** The streaming pipeline now loads trained task classifiers, maintains a rolling feature buffer, predicts tasks per window, and emits predictions alongside VLM captions. See `backend/stream_processor.py` for details.
- [x] Add frontend UI for upload → extract → label → train → evaluate
  **Done:** The frontend UI supports the full ML pipeline workflow, including upload, feature extraction, labeling, training, and evaluation.
- [x] Add temporal models (LSTM/TCN/Transformer) for improved accuracy
  **Done:** The training pipeline now supports LSTM-based temporal models. Use `--model lstm` in `train_task_classifier.py` to train on windowed sequences. (TCN/Transformer can be added similarly.)
- [ ] Add active learning loop: surface low-confidence spans for relabeling
- [ ] Add ablation experiments: MP-only, YOLO-only, fused
- [ ] Add online learning: retrain models incrementally with new data

---

## Requirements

```txt
pandas
pyarrow
scikit-learn
lightgbm
numpy<2
opencv-python
mediapipe
ultralytics
torch
```

Install:

```bash
pip install -r backend/requirements.txt
```

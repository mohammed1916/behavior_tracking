"""Build sliding-window dataset from extracted features for task classification.

Creates train/val/test splits at the video level, applies normalization,
balances classes, and outputs windowed sequences ready for ML training.

Output format:
- X_train.npy: [N_windows, T, F] array of windowed features
- y_train.npy: [N_windows,] array of window labels
- X_val.npy, y_val.npy: validation split
- X_test.npy, y_test.npy: test split
- metadata.json: feature names, normalization params, class mapping

Usage:
    python scripts/build_task_dataset.py --features features.parquet --output dataset/
    python scripts/build_task_dataset.py --features f1.parquet f2.parquet --output dataset/ --window-sec 2.0 --step-sec 0.5
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_and_merge_features(feature_files: List[str]) -> pd.DataFrame:
    """Load multiple feature files and concatenate."""
    dfs = []
    for fpath in feature_files:
        if not os.path.exists(fpath):
            logger.warning(f"Feature file not found: {fpath}")
            continue
        df = pd.read_parquet(fpath)
        dfs.append(df)
        logger.info(f"Loaded {len(df)} rows from {fpath}")
    
    if not dfs:
        raise RuntimeError("No feature files loaded")
    
    df_merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged {len(df_merged)} total rows from {len(dfs)} files")
    
    return df_merged


def extract_numeric_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Extract numeric feature columns (exclude JSON/object columns)."""
    # Select numeric columns only (exclude video_id, frame_index, t, label, JSON columns)
    exclude_cols = ['video_id', 'frame_index', 't', 'label', 
                    'mp_pose_landmarks', 'mp_left_hand_landmarks', 'mp_right_hand_landmarks',
                    'yolo_objects', 'yolo_boxes', 'yolo_confidences', 'class_counts']
    
    numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    logger.info(f"Selected {len(numeric_cols)} numeric feature columns")
    logger.info(f"Features: {numeric_cols[:10]}...")
    
    # Fill NaN with 0 (missing detections)
    X = df[numeric_cols].fillna(0).values
    
    return X, numeric_cols


def create_windows(
    X: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    video_ids: np.ndarray,
    window_sec: float,
    step_sec: float,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from features.
    
    Args:
        X: [N, F] feature array
        labels: [N,] label array
        timestamps: [N,] timestamp array
        video_ids: [N,] video ID array
        window_sec: Window length in seconds
        step_sec: Step size in seconds
        fps: Frames per second (for frame-to-time conversion)
    
    Returns:
        X_windows: [N_windows, T, F] windowed features
        y_windows: [N_windows,] window labels (majority vote)
        video_id_windows: [N_windows,] video ID per window
    """
    window_frames = int(window_sec * fps)
    step_frames = int(step_sec * fps)
    
    if window_frames < 1:
        window_frames = 1
    if step_frames < 1:
        step_frames = 1
    
    logger.info(f"Creating windows: {window_frames} frames ({window_sec}s), step {step_frames} frames ({step_sec}s)")
    
    X_windows = []
    y_windows = []
    video_id_windows = []
    
    # Process each video separately to avoid cross-video windows
    unique_videos = np.unique(video_ids)
    
    for vid in unique_videos:
        vid_mask = video_ids == vid
        X_vid = X[vid_mask]
        labels_vid = labels[vid_mask]
        timestamps_vid = timestamps[vid_mask]
        
        # Sort by timestamp
        sort_idx = np.argsort(timestamps_vid)
        X_vid = X_vid[sort_idx]
        labels_vid = labels_vid[sort_idx]
        
        # Create windows
        for start_idx in range(0, len(X_vid) - window_frames + 1, step_frames):
            end_idx = start_idx + window_frames
            
            X_window = X_vid[start_idx:end_idx]
            labels_window = labels_vid[start_idx:end_idx]
            
            # Skip if window has missing labels
            if np.any(pd.isna(labels_window)):
                continue
            
            # Majority vote for window label
            unique, counts = np.unique(labels_window, return_counts=True)
            window_label = unique[np.argmax(counts)]
            
            X_windows.append(X_window)
            y_windows.append(window_label)
            video_id_windows.append(vid)
    
    X_windows = np.array(X_windows)  # [N_windows, T, F]
    y_windows = np.array(y_windows)
    video_id_windows = np.array(video_id_windows)
    
    logger.info(f"Created {len(X_windows)} windows from {len(unique_videos)} videos")
    
    return X_windows, y_windows, video_id_windows


def balance_classes(X: np.ndarray, y: np.ndarray, video_ids: np.ndarray, method: str = 'downsample') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance classes by downsampling or upsampling.
    
    Args:
        X: [N, T, F] windowed features
        y: [N,] labels
        video_ids: [N,] video IDs
        method: 'downsample' or 'upsample'
    
    Returns:
        X_balanced, y_balanced, video_ids_balanced
    """
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution before balancing: {dict(zip(unique, counts))}")
    
    if method == 'downsample':
        min_count = counts.min()
        indices = []
        for cls in unique:
            cls_idx = np.where(y == cls)[0]
            sampled_idx = np.random.choice(cls_idx, size=min_count, replace=False)
            indices.extend(sampled_idx)
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        X_balanced = X[indices]
        y_balanced = y[indices]
        video_ids_balanced = video_ids[indices]
    
    elif method == 'upsample':
        max_count = counts.max()
        indices = []
        for cls in unique:
            cls_idx = np.where(y == cls)[0]
            sampled_idx = np.random.choice(cls_idx, size=max_count, replace=True)
            indices.extend(sampled_idx)
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        X_balanced = X[indices]
        y_balanced = y[indices]
        video_ids_balanced = video_ids[indices]
    
    else:
        X_balanced, y_balanced, video_ids_balanced = X, y, video_ids
    
    unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
    logger.info(f"Class distribution after balancing: {dict(zip(unique_bal, counts_bal))}")
    
    return X_balanced, y_balanced, video_ids_balanced


def split_by_video(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset by video ID to prevent leakage.
    
    Args:
        X: [N, T, F] windowed features
        y: [N,] labels
        video_ids: [N,] video IDs
        train_ratio: proportion for training
        val_ratio: proportion for validation
        test_ratio: proportion for testing
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    unique_videos = np.unique(video_ids)
    logger.info(f"Splitting {len(unique_videos)} videos: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Split videos
    train_videos, temp_videos = train_test_split(unique_videos, train_size=train_ratio, random_state=42)
    val_videos, test_videos = train_test_split(temp_videos, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
    
    # Split windows
    train_mask = np.isin(video_ids, train_videos)
    val_mask = np.isin(video_ids, val_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train: {len(X_train)} windows from {len(train_videos)} videos")
    logger.info(f"Val: {len(X_val)} windows from {len(val_videos)} videos")
    logger.info(f"Test: {len(X_test)} windows from {len(test_videos)} videos")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Normalize features using StandardScaler fitted on training set.
    
    Args:
        X_train: [N_train, T, F]
        X_val: [N_val, T, F]
        X_test: [N_test, T, F]
    
    Returns:
        X_train_norm, X_val_norm, X_test_norm, scaler
    """
    # Flatten windows for fitting scaler
    N_train, T, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    # Transform all splits
    X_train_norm = scaler.transform(X_train.reshape(-1, F)).reshape(N_train, T, F)
    
    if len(X_val) > 0:
        N_val = len(X_val)
        X_val_norm = scaler.transform(X_val.reshape(-1, F)).reshape(N_val, T, F)
    else:
        X_val_norm = X_val
    
    if len(X_test) > 0:
        N_test = len(X_test)
        X_test_norm = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, T, F)
    else:
        X_test_norm = X_test
    
    logger.info(f"Normalized features: mean={scaler.mean_[:5]}, std={scaler.scale_[:5]}")
    
    return X_train_norm, X_val_norm, X_test_norm, scaler


def build_dataset(
    feature_files: List[str],
    output_dir: str,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    balance_method: str = 'downsample',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """Build complete windowed dataset from feature files.
    
    Args:
        feature_files: List of Parquet feature file paths
        output_dir: Directory to save dataset
        window_sec: Window length in seconds
        step_sec: Step size in seconds
        balance_method: 'downsample', 'upsample', or 'none'
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load features
    df = load_and_merge_features(feature_files)
    
    # Filter for labeled frames only
    if 'label' not in df.columns or not df['label'].notna().any():
        raise RuntimeError("No labeled frames found in feature files")
    
    df_labeled = df[df['label'].notna()].copy()
    logger.info(f"Using {len(df_labeled)} labeled frames out of {len(df)} total")
    
    # Extract numeric features
    X, feature_names = extract_numeric_features(df_labeled)
    labels = df_labeled['label'].values
    timestamps = df_labeled['t'].values
    video_ids = df_labeled['video_id'].values
    
    # Estimate FPS from first video
    first_vid = video_ids[0]
    vid_mask = video_ids == first_vid
    vid_times = timestamps[vid_mask]
    vid_frames = df_labeled[vid_mask]['frame_index'].values
    fps = len(vid_frames) / (vid_times.max() - vid_times.min() + 0.001) if len(vid_times) > 1 else 30.0
    logger.info(f"Estimated FPS: {fps:.2f}")
    
    # Create windows
    X_windows, y_windows, video_id_windows = create_windows(
        X, labels, timestamps, video_ids, window_sec, step_sec, fps
    )
    
    # Balance classes
    if balance_method != 'none':
        X_windows, y_windows, video_id_windows = balance_classes(
            X_windows, y_windows, video_id_windows, method=balance_method
        )
    
    # Split by video
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_video(
        X_windows, y_windows, video_id_windows, train_ratio, val_ratio, test_ratio
    )
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)
    
    # Save arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_norm)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_norm)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_norm)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metadata
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    metadata = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'window_sec': window_sec,
        'step_sec': step_sec,
        'fps': fps,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'label_to_idx': label_to_idx,
        'idx_to_label': {idx: label for label, idx in label_to_idx.items()},
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'train_label_dist': dict(zip(*np.unique(y_train, return_counts=True))),
        'val_label_dist': dict(zip(*np.unique(y_val, return_counts=True))),
        'test_label_dist': dict(zip(*np.unique(y_test, return_counts=True))),
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset saved to {output_dir}")
    logger.info(f"Train: {len(X_train)} windows, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Window shape: {X_train.shape}")
    logger.info(f"Labels: {list(label_to_idx.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Build windowed dataset from extracted features')
    parser.add_argument('--features', nargs='+', required=True, help='Parquet feature file(s)')
    parser.add_argument('--output', required=True, help='Output directory for dataset')
    parser.add_argument('--window-sec', type=float, default=2.0, help='Window length in seconds (default: 2.0)')
    parser.add_argument('--step-sec', type=float, default=0.5, help='Step size in seconds (default: 0.5)')
    parser.add_argument('--balance', choices=['downsample', 'upsample', 'none'], default='downsample',
                        help='Class balancing method (default: downsample)')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    build_dataset(
        feature_files=args.features,
        output_dir=args.output,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        balance_method=args.balance,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == '__main__':
    main()

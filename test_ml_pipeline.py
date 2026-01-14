"""Test the ML training pipeline end-to-end with a small synthetic dataset."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.extract_features import extract_features_from_video
from scripts.build_task_dataset import build_dataset
from scripts.train_task_classifier import train_classifier


def create_synthetic_video(output_path: str, num_frames: int = 120, fps: float = 30.0):
    """Create a synthetic video for testing (person moving hands)."""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create frame with moving rectangle (simulating a hand)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add "hand" that moves in first half (work), stays still in second half (idle)
        if i < num_frames // 2:
            # Work: moving hand
            x = int(width * 0.3 + (width * 0.4) * (i / (num_frames / 2)))
            y = height // 2
        else:
            # Idle: stationary hand
            x = int(width * 0.7)
            y = height // 2
        
        cv2.rectangle(frame, (x-30, y-30), (x+30, y+30), (255, 0, 0), -1)
        
        # Add some "noise" to make it more realistic
        cv2.circle(frame, (width//4, height//4), 20, (0, 255, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created synthetic video: {output_path} ({num_frames} frames)")


def create_synthetic_labels(output_path: str, num_frames: int = 120):
    """Create synthetic labels CSV."""
    labels = []
    for i in range(num_frames):
        label = 'work' if i < num_frames // 2 else 'idle'
        labels.append({'frame_index': i, 'label': label})
    
    df = pd.DataFrame(labels)
    df.to_csv(output_path, index=False)
    print(f"Created synthetic labels: {output_path} ({len(labels)} frames)")


def test_pipeline():
    """Test the full ML training pipeline."""
    print("\n" + "="*80)
    print("Testing ML Training Pipeline")
    print("="*80 + "\n")
    
    # Create temporary directory for test files
    test_dir = tempfile.mkdtemp(prefix='ml_pipeline_test_')
    print(f"Test directory: {test_dir}\n")
    
    try:
        # 1. Create synthetic data
        print("STAGE 1: Creating synthetic video and labels")
        print("-" * 80)
        
        video_path = os.path.join(test_dir, 'test_video.mp4')
        labels_path = os.path.join(test_dir, 'test_labels.csv')
        
        create_synthetic_video(video_path, num_frames=120, fps=30.0)
        create_synthetic_labels(labels_path, num_frames=120)
        
        # 2. Extract features
        print("\nSTAGE 2: Extracting features")
        print("-" * 80)
        
        features_path = os.path.join(test_dir, 'test_features.parquet')
        
        extract_features_from_video(
            video_path=video_path,
            output_path=features_path,
            video_id='test_video',
            labels_csv=labels_path,
            detector_type='fusion',  # Use fusion for full feature set
            sample_rate=1,
            device='cpu',  # Use CPU for testing
        )
        
        print(f"\nFeatures saved to: {features_path}")
        
        # Verify features
        df_features = pd.read_parquet(features_path)
        print(f"Feature shape: {df_features.shape}")
        print(f"Columns: {list(df_features.columns)[:10]}...")
        print(f"Label distribution:\n{df_features['label'].value_counts()}")
        
        # 3. Build dataset
        print("\nSTAGE 3: Building windowed dataset")
        print("-" * 80)
        
        dataset_dir = os.path.join(test_dir, 'test_dataset')
        
        build_dataset(
            feature_files=[features_path],
            output_dir=dataset_dir,
            window_sec=1.0,  # Shorter window for small test
            step_sec=0.5,
            balance_method='none',  # Don't balance for small test
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        
        print(f"\nDataset saved to: {dataset_dir}")
        
        # Verify dataset
        X_train = np.load(os.path.join(dataset_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(dataset_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
        
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
        
        # 4. Train model
        print("\nSTAGE 4: Training classifier")
        print("-" * 80)
        
        model_path = os.path.join(test_dir, 'test_model.pkl')
        
        metrics = train_classifier(
            dataset_dir=dataset_dir,
            model_type='rf',
            output_path=model_path,
            aggregation='stats',
            n_estimators=10,  # Fewer trees for fast testing
        )
        
        print(f"\nModel saved to: {model_path}")
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # 5. Summary
        print("\n" + "="*80)
        print("PIPELINE TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  Video: {video_path}")
        print(f"  Labels: {labels_path}")
        print(f"  Features: {features_path}")
        print(f"  Dataset: {dataset_dir}")
        print(f"  Model: {model_path}")
        print(f"\nTest accuracy: {metrics['accuracy']:.1%}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\nCleaned up test directory: {test_dir}")
        except Exception:
            print(f"\nWarning: Failed to clean up {test_dir}")


if __name__ == '__main__':
    success = test_pipeline()
    sys.exit(0 if success else 1)

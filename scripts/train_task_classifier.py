"""Train baseline task classifier on windowed features.

Trains RandomForest or LightGBM on aggregated window statistics,
evaluates on test set, and saves model to mlruns/.

Aggregation strategies:
- 'stats': mean, std, min, max per feature across window
- 'flatten': flatten entire window into 1D vector (T*F features)

Usage:
    python scripts/train_task_classifier.py --dataset dataset/ --model rf --output model.pkl
    python scripts/train_task_classifier.py --dataset dataset/ --model lgbm --agg flatten --output model.pkl
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import lightgbm as lgb

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def aggregate_windows(X: np.ndarray, method: str = 'stats') -> np.ndarray:
    """Aggregate window features for non-temporal models.
    
    Args:
        X: [N, T, F] windowed features
        method: 'stats' (mean/std/min/max) or 'flatten'
    
    Returns:
        X_agg: [N, F_agg] aggregated features
    """
    N, T, F = X.shape
    
    if method == 'stats':
        # Compute stats across time dimension
        mean = X.mean(axis=1)  # [N, F]
        std = X.std(axis=1)    # [N, F]
        min_val = X.min(axis=1)
        max_val = X.max(axis=1)
        
        X_agg = np.concatenate([mean, std, min_val, max_val], axis=1)  # [N, 4*F]
        logger.info(f"Aggregated windows: {X.shape} -> {X_agg.shape} (stats)")
    
    elif method == 'flatten':
        # Flatten window into 1D
        X_agg = X.reshape(N, T * F)
        logger.info(f"Aggregated windows: {X.shape} -> {X_agg.shape} (flatten)")
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return X_agg


def load_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load train/val/test splits and metadata."""
    X_train = np.load(os.path.join(dataset_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(dataset_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(dataset_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(dataset_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(dataset_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(dataset_dir, 'y_test.npy'))
    
    with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded dataset from {dataset_dir}")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=kwargs.get('n_estimators', 100),
        max_depth=kwargs.get('max_depth', None),
        min_samples_split=kwargs.get('min_samples_split', 2),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    
    logger.info("Training RandomForest...")
    model.fit(X_train, y_train)
    logger.info("Training complete")
    
    return model


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> lgb.LGBMClassifier:
    """Train LightGBM classifier."""
    model = lgb.LGBMClassifier(
        n_estimators=kwargs.get('n_estimators', 100),
        max_depth=kwargs.get('max_depth', -1),
        learning_rate=kwargs.get('learning_rate', 0.1),
        num_leaves=kwargs.get('num_leaves', 31),
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    
    logger.info("Training LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)],
    )
    logger.info("Training complete")
    
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, metadata: Dict) -> Dict:
    """Evaluate model on test set and return metrics."""
    logger.info("Evaluating on test set...")
    
    y_pred = model.predict(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")
    
    # Per-class report
    label_names = [metadata['idx_to_label'].get(str(i), str(i)) for i in sorted(set(y_test))]
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics


def save_model(model, output_path: str, metadata: Dict, metrics: Dict, aggregation: str):
    """Save model with metadata and metrics."""
    model_data = {
        'model': model,
        'metadata': metadata,
        'metrics': metrics,
        'aggregation': aggregation,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {output_path}")
    
    # Save metrics separately as JSON for easy inspection
    metrics_path = output_path.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")


def train_classifier(
    dataset_dir: str,
    model_type: str,
    output_path: str,
    aggregation: str = 'stats',
    **model_kwargs
):
    """Train task classifier pipeline.
    
    Args:
        dataset_dir: Path to dataset directory (with X_train.npy, etc.)
        model_type: 'rf' (RandomForest) or 'lgbm' (LightGBM)
        output_path: Path to save trained model
        aggregation: 'stats' or 'flatten'
        **model_kwargs: Additional model hyperparameters
    """
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = load_dataset(dataset_dir)
    
    # Aggregate windows
    X_train_agg = aggregate_windows(X_train, method=aggregation)
    X_val_agg = aggregate_windows(X_val, method=aggregation)
    X_test_agg = aggregate_windows(X_test, method=aggregation)
    
    # Train model
    if model_type == 'rf':
        model = train_random_forest(X_train_agg, y_train, **model_kwargs)
    elif model_type == 'lgbm':
        model = train_lightgbm(X_train_agg, y_train, X_val_agg, y_val, **model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate
    metrics = evaluate_model(model, X_test_agg, y_test, metadata)
    
    # Save
    save_model(model, output_path, metadata, metrics, aggregation)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train baseline task classifier')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', choices=['rf', 'lgbm'], default='rf', help='Model type (default: rf)')
    parser.add_argument('--output', required=True, help='Path to save trained model (.pkl)')
    parser.add_argument('--agg', choices=['stats', 'flatten'], default='stats',
                        help='Aggregation method (default: stats)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=None, help='Max tree depth (default: None)')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate for LightGBM (default: 0.1)')
    
    args = parser.parse_args()
    
    model_kwargs = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
    }
    
    if args.model == 'lgbm':
        model_kwargs['learning_rate'] = args.learning_rate
    
    train_classifier(
        dataset_dir=args.dataset,
        model_type=args.model,
        output_path=args.output,
        aggregation=args.agg,
        **model_kwargs
    )


if __name__ == '__main__':
    main()

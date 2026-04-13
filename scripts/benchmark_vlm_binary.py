"""Benchmark a local VLM on binary work/idle labels.

This script:
- loads raw videos from the Assembly dataset
- samples frames at a fixed interval
- captions each frame with a local VLM
- maps the caption to binary work/idle using backend.rules.normalize_label_text
- compares predictions to generated label CSVs
- writes metrics and sample predictions to a JSON log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from backend.rules import normalize_label_text


def load_label_map(label_csv: Path) -> Dict[int, str]:
    with label_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(row["frame_index"]): row["label"] for row in reader}


def list_video_ids(labels_dir: Path, max_videos: int | None) -> List[str]:
    ids = []
    for path in sorted(labels_dir.glob("*_labels.csv")):
        ids.append(path.name.replace("_labels.csv", ""))
    if max_videos is not None:
        ids = ids[:max_videos]
    return ids


def make_pipeline(model_id: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model=model_id, device=device), device


def caption_frame(pipe, frame_bgr) -> str:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    out = pipe(pil)
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return str(out[0]["generated_text"])
    return str(out)


def benchmark_video(
    video_path: Path,
    labels: Dict[int, str],
    pipe,
    sample_every: int,
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    y_true: List[str] = []
    y_pred: List[str] = []
    samples: List[Dict[str, object]] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for frame_idx in range(0, frame_count, sample_every):
        gt = labels.get(frame_idx)
        if gt not in {"work", "idle"}:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        caption = caption_frame(pipe, frame)
        pred = normalize_label_text(caption, output_mode="binary")
        y_true.append(gt)
        y_pred.append(pred)
        if len(samples) < 10:
            samples.append(
                {
                    "frame_index": frame_idx,
                    "ground_truth": gt,
                    "prediction": pred,
                    "caption": caption,
                }
            )
    cap.release()
    return y_true, y_pred, samples


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    labels = ["idle", "work"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_idle": float(precision[0]),
        "recall_idle": float(recall[0]),
        "f1_idle": float(f1[0]),
        "support_idle": int(support[0]),
        "precision_work": float(precision[1]),
        "recall_work": float(recall[1]),
        "f1_work": float(f1[1]),
        "support_work": int(support[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BLIP/Qwen VLM on work/idle labels")
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--sample-every", type=int, default=60, help="Sample every N frames")
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    labels_dir = Path(args.labels_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    pipe, device = make_pipeline(args.model)

    all_true: List[str] = []
    all_pred: List[str] = []
    per_video: List[Dict[str, object]] = []

    for video_id in list_video_ids(labels_dir, args.max_videos):
        video_path = videos_dir / f"{video_id}.avi"
        label_path = labels_dir / f"{video_id}_labels.csv"
        if not video_path.exists() or not label_path.exists():
            continue

        labels = load_label_map(label_path)
        y_true, y_pred, sample_rows = benchmark_video(
            video_path=video_path,
            labels=labels,
            pipe=pipe,
            sample_every=args.sample_every,
        )
        if not y_true:
            continue

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        per_video.append(
            {
                "video_id": video_id,
                "num_samples": len(y_true),
                "metrics": compute_metrics(y_true, y_pred),
                "samples": sample_rows,
            }
        )

    if not all_true:
        raise SystemExit("No benchmark samples were collected")

    result = {
        "benchmark": "vlm_binary_work_idle",
        "model": args.model,
        "device": str(device),
        "videos_dir": str(videos_dir),
        "labels_dir": str(labels_dir),
        "sample_every_frames": args.sample_every,
        "max_videos": args.max_videos,
        "num_videos_evaluated": len(per_video),
        "num_samples_total": len(all_true),
        "runtime_sec": round(time.time() - started, 2),
        "metrics": compute_metrics(all_true, all_pred),
        "per_video": per_video,
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

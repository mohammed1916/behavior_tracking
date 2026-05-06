"""Convert procedure_anchors.csv to frame-level labels.

Supports:
- binary labels: frames inside procedure anchors -> work, outside -> idle
- procedure labels: frames inside procedure anchors -> procedure_<n>, outside -> idle
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple


AnchorSegment = Tuple[int, int, int]


def load_anchor_segments(csv_path: str, video_id: str) -> List[AnchorSegment]:
    """Load procedure anchor segments for a specific video id.

    CSV format is:
    - first column: video id
    - remaining columns: alternating start/end frame bounds for procedure 0..N
      Example row:
      video, 0,136, 136,238, 238,339, ...
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)

        for row in reader:
            if not row or row[0].strip() != video_id:
                continue

            values = [cell.strip() for cell in row[1:] if cell.strip() != ""]
            segments: List[AnchorSegment] = []
            proc_num = 0
            for i in range(0, len(values), 2):
                if i + 1 >= len(values):
                    break
                start = int(values[i])
                end = int(values[i + 1])
                if end < start:
                    start, end = end, start
                segments.append((proc_num, start, end))
                proc_num += 1
            return segments

    return []


def build_frame_labels(
    segments: List[AnchorSegment],
    mode: str = "binary",
) -> List[Dict[str, str]]:
    """Build per-frame labels from procedure segments.

    Rule for binary mode:
    - frame inside any procedure segment => work
    - frame outside all procedure segments => idle
    """
    if not segments:
        return []

    max_frame = max(end for _, _, end in segments)
    frame_labels = ["idle"] * (max_frame + 1)

    for proc_num, start, end in segments:
        label = "work" if mode == "binary" else f"procedure_{proc_num}"
        for frame_idx in range(start, end + 1):
            frame_labels[frame_idx] = label

    return [
        {"frame_index": frame_idx, "label": label}
        for frame_idx, label in enumerate(frame_labels)
    ]


def parse_procedure_anchors(csv_path: str, video_id: str, mode: str = "binary") -> List[Dict[str, str]]:
    segments = load_anchor_segments(csv_path, video_id)
    return build_frame_labels(segments, mode=mode)


def export_all_labels(csv_path: str, output_dir: str, mode: str = "binary") -> int:
    """Export one label CSV per video id found in the anchors file."""
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)

        for row in reader:
            if not row or not row[0].strip():
                continue
            video_id = row[0].strip()
            labels = parse_procedure_anchors(csv_path, video_id, mode=mode)
            if not labels:
                continue

            output_path = os.path.join(output_dir, f"{video_id}_labels.csv")
            with open(output_path, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=["frame_index", "label"])
                writer.writeheader()
                writer.writerows(labels)
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert procedure anchors into frame-level labels")
    parser.add_argument("csv_path", help="Path to procedure_anchors.csv")
    parser.add_argument("--video-id", help="Specific video id to export")
    parser.add_argument("--output", required=True, help="Output CSV file or directory")
    parser.add_argument(
        "--mode",
        choices=["binary", "procedure"],
        default="binary",
        help="binary: work/idle, procedure: procedure_N/idle",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export labels for all video ids into the output directory",
    )
    args = parser.parse_args()

    if args.all:
        count = export_all_labels(args.csv_path, args.output, mode=args.mode)
        print(f"Wrote {count} label CSV files to {args.output}")
        return

    if not args.video_id:
        raise SystemExit("--video-id is required unless --all is used")

    labels = parse_procedure_anchors(args.csv_path, args.video_id, mode=args.mode)
    if not labels:
        raise SystemExit(f"No labels found for video: {args.video_id}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "label"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Wrote {len(labels)} frame labels to {args.output}")


if __name__ == "__main__":
    main()

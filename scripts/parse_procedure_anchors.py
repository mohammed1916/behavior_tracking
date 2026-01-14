"""Convert procedure_anchors.csv to frame-level labels for ML training."""

import csv
import sys

def parse_procedure_anchors(csv_path: str, video_id: str) -> list:
    """Parse procedure_anchors.csv and extract labels for a specific video.
    
    Format: File name, [0, start, end, 1, start, end, ..., 8, start, end]
    Each pair (start, end) marks when procedure N is active.
    """
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if not row or row[0] != video_id:
                continue
            
            filename = row[0]
            # Parse procedure segments: alternating procedure_num, start, end
            procedures = []
            for i in range(1, len(row), 2):
                if i + 1 < len(row):
                    try:
                        proc_num = int(row[i])
                        start = int(row[i + 1])
                        end = int(row[i + 2]) if i + 2 < len(row) else int(row[i + 1])
                        procedures.append((proc_num, start, end))
                    except (ValueError, IndexError):
                        pass
            
            # Create frame-level labels
            max_frame = max([end for _, _, end in procedures]) if procedures else 1000
            frame_labels = ['work'] * (max_frame + 1)  # Default all to work (assembly activity)
            
            # Mark procedure transitions as idle (brief pauses between procedures)
            for i, (proc_num, start, end) in enumerate(procedures):
                # Mark this procedure segment
                for frame in range(start, min(end + 1, max_frame + 1)):
                    frame_labels[frame] = f'procedure_{proc_num}'
            
            # Convert to output format: frame_index, label
            for frame_idx, label in enumerate(frame_labels):
                labels.append({'frame_index': frame_idx, 'label': label})
            
            return labels
    
    return []


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python scripts/parse_procedure_anchors.py <csv_path> <video_id> <output_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    video_id = sys.argv[2]
    output_path = sys.argv[3]
    
    labels = parse_procedure_anchors(csv_path, video_id)
    
    if not labels:
        print(f"No labels found for video: {video_id}")
        sys.exit(1)
    
    # Write output CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_index', 'label'])
        writer.writeheader()
        writer.writerows(labels)
    
    print(f"Wrote {len(labels)} frame labels to {output_path}")

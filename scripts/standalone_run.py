"""Standalone runner to list local VLM models and caption a video using a chosen model.

Usage examples:
  python standalone_run.py --list
  python standalone_run.py --video path/to/video.mp4
  python standalone_run.py --video path/to/video.mp4 --model 0
  python standalone_run.py --video path/to/video.mp4 --model "Salesforce/blip-image-captioning-large" --out result.json

The script reads `../local_vlm_models.json` to discover available models.
It extracts a representative frame (middle) from the video and runs an `image-to-text`
pipeline from `transformers` to produce a caption.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def load_models(models_file: Path):
    if not models_file.exists():
        return []
    try:
        data = json.loads(models_file.read_text())
        return data.get('models', [])
    except Exception:
        return []


def list_models(models):
    if not models:
        print('No models found in local_vlm_models.json')
        return
    for i, m in enumerate(models):
        mid = m.get('id') or m.get('name') or str(i)
        name = m.get('name', '')
        task = m.get('task', '')
        print(f'[{i}] {mid} -- {name} -- task={task}')


def pick_model(models, selector):
    if selector is None:
        return models[0]['id'] if models else None
    # If selector is an integer index
    try:
        idx = int(selector)
        if 0 <= idx < len(models):
            return models[idx].get('id')
    except Exception:
        pass
    # Otherwise treat selector as id string
    for m in models:
        if m.get('id') == selector or m.get('name') == selector:
            return m.get('id')
    # fallback: return selector as-is
    return selector


def caption_video_frame(video_path, model_id):
    import cv2
    from PIL import Image
    import numpy as np

    res = {'video_path': str(video_path), 'model': model_id}
    if not os.path.exists(video_path):
        res['error'] = 'video not found'
        return res

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        res['error'] = 'could not open video'
        return res

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    res.update({'fps': float(fps), 'frame_count': frame_count, 'width': width, 'height': height})

    mid = frame_count // 2 if frame_count > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if not ret or frame is None:
        res['warning'] = 'could not read frame'
        return res

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    caption = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        p = pipeline('image-to-text', model=model_id, device=device)
        out = p(pil)
        if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
            caption = out[0]['generated_text']
        else:
            caption = str(out)
    except Exception as e:
        res['caption_error'] = str(e)

    avg_color_per_row = frame.mean(axis=0).mean(axis=0)
    avg_color = [float(avg_color_per_row[2]), float(avg_color_per_row[1]), float(avg_color_per_row[0])]
    res['first_frame_avg_color_bgr'] = avg_color
    res['caption'] = caption
    res['duration_sec'] = frame_count / fps if fps > 0 else 0

    return res


def main():
    parser = argparse.ArgumentParser(description='Standalone VLM video captioning runner')
    parser.add_argument('--list', action='store_true', help='List available local VLM models')
    parser.add_argument('--video', type=str, help='Path to video file to caption')
    parser.add_argument('--model', type=str, help='Model index or id to use (defaults to first model)')
    parser.add_argument('--out', type=str, help='Optional output JSON file to write results')

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models_file = repo_root / 'local_vlm_models.json'
    models = load_models(models_file)

    if args.list:
        list_models(models)
        sys.exit(0)

    if not args.video:
        # print('Provide --video path or use --list to see available models', file=sys.stderr)
        print('Using default Path:', repo_root / 'backend' / 'data'/ 'assembly_idle.mp4' )
        args.video = str( repo_root / 'backend' / 'data'/ 'assembly_idle.mp4' )
        sys.exit(2)

    model_id = pick_model(models, args.model)
    if model_id is None:
        print('No model selected and no models available', file=sys.stderr)
        sys.exit(3)

    result = caption_video_frame(args.video, model_id)
    out_json = json.dumps(result, indent=2)
    print(out_json)
    if args.out:
        try:
            Path(args.out).write_text(out_json)
        except Exception as e:
            print(f'Failed to write out file: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()

import json
import sys
import os
import cv2

video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join('..', 'data', 'drone_assembly.mp4')
video_path = os.path.abspath(video_path)
res = {'video_path': video_path}

if not os.path.exists(video_path):
    res['error'] = 'video not found'
    print(json.dumps(res))
    sys.exit(1)

# Basic video analysis
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    res['error'] = 'could not open video'
    print(json.dumps(res))
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
res.update({'fps': float(fps), 'frame_count': frame_count, 'width': width, 'height': height})

# Extract middle frame
mid = frame_count // 2 if frame_count > 0 else 0
cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
ret, frame = cap.read()
if not ret or frame is None:
    res['warning'] = 'could not read frame'
    print(json.dumps(res))
    sys.exit(0)

# Convert to RGB PIL
from PIL import Image
import numpy as np
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil = Image.fromarray(rgb)

# Run local BLIP pipeline, forcing offline mode so it uses cached files
caption = None
try:
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    from transformers import pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    p = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large', device=device)
    out = p(pil)
    if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
        caption = out[0]['generated_text']
    else:
        caption = str(out)
except Exception as e:
    res['caption_error'] = str(e)

# Average color of the frame
avg_color_per_row = frame.mean(axis=0).mean(axis=0)
avg_color = [float(avg_color_per_row[2]), float(avg_color_per_row[1]), float(avg_color_per_row[0])]
res['first_frame_avg_color_bgr'] = avg_color
res['caption'] = caption
res['duration_sec'] = frame_count / fps if fps > 0 else 0

print(json.dumps(res, indent=2))

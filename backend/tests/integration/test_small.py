"""Quick test with smaller video file."""
import urllib.request
import json
import time

video_path = r"C:\Users\BBBS-AI-01\d\behavior_tracking\data\drone_assembly.mp4"

import os
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at {video_path}")
    exit(1)

print(f"Uploading: {os.path.basename(video_path)} ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)")

# Upload
import io
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = io.BytesIO()

with open(video_path, 'rb') as f:
    video_data = f.read()

body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="video"; filename="test.mp4"\r\n')
body.write(b'Content-Type: video/mp4\r\n\r\n')
body.write(video_data)
body.write(f'\r\n--{boundary}--\r\n'.encode())

req = urllib.request.Request(
    'http://localhost:8001/backend/upload_vlm',
    data=body.getvalue(),
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
        filename = result['filename']
        print(f"✓ Uploaded as: {filename}\n")
except Exception as e:
    print(f"✗ Upload failed: {e}")
    exit(1)

# Stream
url = (
    f"http://localhost:8001/backend/vlm_local_stream?"
    f"filename={filename}"
    f"&model=qwen/qwen2-vl-2b-instruct"
    f"&prompt=What+activity"
    f"&classifier_source=llm"
    f"&classifier_mode=multi"
)

print(f"Streaming LLM mode...\n")

events = []
segments = []
samples = []

try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            line_str = line.decode('utf-8').strip()
            if not line_str or not line_str.startswith('data: '):
                continue
            
            try:
                data = json.loads(line_str[6:])
                events.append(data)
                stage = data.get('stage', '')
                
                if stage == 'started':
                    print(f"[stream] Started")
                elif stage == 'video_info':
                    print(f"[info] {data.get('frame_count', 0)} frames @ {data.get('fps', 0):.0f}fps")
                elif stage == 'sample':
                    samples.append(data)
                    if len(samples) % 5 == 1:
                        print(f"[sample {len(samples)}] {data.get('label', '?')} - {data.get('caption', '')[:50]}")
                elif stage == 'segment':
                    segments.append(data)
                    print(f"[segment {len(segments)}] {data.get('label', '?')} ({data.get('duration', 0):.1f}s)")
                    print(f"         timeline: {data.get('timeline', '')[:100]}")
                    if data.get('llm_output'):
                        print(f"         llm: {data.get('llm_output', '')[:80]}")
                elif stage == 'finished':
                    print(f"[done] {data.get('message', '')}")
                    break
            except json.JSONDecodeError:
                pass
                
except Exception as e:
    print(f"Error: {e}")

print(f"\n{'='*60}")
print(f"Results: {len(events)} events, {len(samples)} samples, {len(segments)} segments")
print(f"{'='*60}")

if segments:
    print("✓ SUCCESS - Segments emitted in LLM mode!")
else:
    print("✗ FAIL - No segments (expected > 0 for LLM mode)")

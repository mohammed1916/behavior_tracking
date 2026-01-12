#!/usr/bin/env python3
"""Quick test to run a video with YOLO detector enabled and check logs"""

import requests
import json
import time

# Pick a small test video
filename = "test.mp4"  # Match one from vlm_uploads
model = "qwen2_vl_7b_instruct"

# Create a proper test URL
url = f"http://localhost:8001/backend/vlm_local_stream"
params = {
    'filename': filename,
    'model': model,
    'prompt': 'Is this work or idle?',
    'classifier_source': 'llm',
    'enable_yolo': 'true',  # Enable YOLO detector
    'enable_mediapipe': 'false',  # Disable MediaPipe to isolate test
}

print(f"Testing with URL: {url}")
print(f"Parameters: {params}")
print("\nStarting stream...")

try:
    response = requests.get(url, params=params, stream=True, timeout=300)
    print(f"Response status: {response.status_code}")
    
    event_count = 0
    sample_count = 0
    detector_count = 0
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            event_count += 1
            
            if line.startswith('data:'):
                try:
                    data = json.loads(line[5:].strip())
                    if data.get('stage') == 'sample':
                        sample_count += 1
                        if 'detector' in data:
                            detector_count += 1
                            if sample_count <= 3 or sample_count % 10 == 0:
                                print(f"  Frame {data.get('frame_index')}: {data['label']}, detector={data['detector'].get('detector_label')}")
                        elif sample_count % 10 == 0:
                            print(f"  Frame {data.get('frame_index')}: {data['label']} (NO DETECTOR)")
                except json.JSONDecodeError:
                    pass
                    
        if event_count % 100 == 0:
            print(f"... processed {event_count} events, {sample_count} samples, {detector_count} with detector")
    
    print(f"\n Stream completed!")
    print(f"Total events: {event_count}")
    print(f"Total samples: {sample_count}")
    print(f"Samples with detector data: {detector_count}")
    print(f"Detector coverage: {100*detector_count/max(1, sample_count):.1f}%")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

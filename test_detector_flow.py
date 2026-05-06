#!/usr/bin/env python3
"""Test script to verify detector metadata flow through API"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

# Find a test video
test_videos = list(Path("dataset/sequential_human_assembly").glob("**/*.mp4"))
if not test_videos:
    test_videos = list(Path("data").glob("**/*.mp4"))

if not test_videos:
    print("No test videos found!")
    exit(1)

test_video = test_videos[0]
print(f"Using test video: {test_video}")

# Upload video
print("\n1. Uploading video...")
with open(test_video, 'rb') as f:
    files = {'video': f}
    resp = requests.post(f"{BASE_URL}/backend/upload_vlm", files=files)
    
if resp.status_code != 200:
    print(f"Upload failed: {resp.status_code}")
    print(resp.text)
    exit(1)

filename = resp.json()['filename']
print(f"   Uploaded as: {filename}")

# Process with detector enabled
print("\n2. Processing video with YOLO detector enabled...")
url = f"{BASE_URL}/backend/vlm_local_stream?filename={filename}&model=blip_base&prompt=What%20is%20happening&enable_yolo=true&classifier_source=caption"

print(f"   URL: {url}")
print("   Streaming response...")

response = requests.get(url, stream=True)
if response.status_code != 200:
    print(f"Processing failed: {response.status_code}")
    print(response.text)
    exit(1)

sample_count = 0
detector_sample_count = 0
analysis_id = None

for line in response.iter_lines():
    if not line:
        continue
    
    try:
        if line.startswith(b'data: '):
            data = json.loads(line[6:])
            
            if data.get('stage') == 'complete':
                analysis_id = data.get('analysis_id')
                print(f"\n   Processing complete! Analysis ID: {analysis_id}")
            elif data.get('stage') == 'sample':
                sample_count += 1
                if 'detector' in data:
                    detector_sample_count += 1
                    print(f"   Frame {sample_count}: detector data present")
                    if sample_count <= 3:  # Show first few
                        print(f"      {json.dumps(data['detector'], indent=8)}")
    except Exception as e:
        pass  # Skip non-JSON lines

print(f"\n3. Results:")
print(f"   Total samples: {sample_count}")
print(f"   Samples with detector data: {detector_sample_count}")

if analysis_id:
    print(f"\n4. Checking database for analysis {analysis_id}...")
    time.sleep(1)  # Wait a moment for db write
    
    # Query the analysis to check detector metadata
    resp = requests.get(f"{BASE_URL}/backend/analysis/{analysis_id}")
    if resp.status_code == 200:
        analysis = resp.json()
        db_detector_count = sum(1 for s in analysis.get('samples', []) if s.get('detector_metadata'))
        print(f"   Samples with detector_metadata in DB: {db_detector_count}")
        
        if db_detector_count > 0:
            print(f"   ✓ SUCCESS: Detector metadata was saved to database!")
            # Show first sample
            for s in analysis.get('samples', []):
                if s.get('detector_metadata'):
                    print(f"\n   Sample {s['frame_index']} detector_metadata:")
                    print(f"      {json.dumps(s['detector_metadata'], indent=8, default=str)[:500]}")
                    break
        else:
            print(f"   ✗ FAILURE: No detector metadata in database")
    else:
        print(f"   Error fetching analysis: {resp.status_code}")

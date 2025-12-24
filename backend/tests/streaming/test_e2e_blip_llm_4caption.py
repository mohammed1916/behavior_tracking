#!/usr/bin/env python3
"""
End-to-end test: VLM+LLM streaming with BLIP model and 4-caption windows.

This test uploads a video and processes it through the `/backend/vlm_local_stream`
endpoint with the BLIP captioner and LLM-based segmentation (4 captions per window).

Usage:
    python -m pytest backend/tests/streaming/test_e2e_blip_llm_4caption.py -v -s

Or run standalone:
    python backend/tests/streaming/test_e2e_blip_llm_4caption.py
    
Output:
    python backend/tests/streaming/test_e2e_blip_llm_4caption.py
        [1] Uploading video...
        ✓ Uploaded as: c534ae6f-6e62-458e-b0be-4394466eb1fe_assembly_idle (2) (2).mp4

        [2] Processing video with BLIP + LLM (4-caption windows)...
        Video: 1929 frames @ 30.0 fps, 64.3s duration
        [1] t=0.00s label=None caption=arafed image of a man working in a factory with a machine...
        [2] t=2.00s label=None caption=someone is working on a laptop with a lot of tools...
        [3] t=4.00s label=None caption=arafed worker working on a machine in a factory...
        [4] t=6.00s label=None caption=arafed man standing in a factory with a camera in his hand...
        [5] t=8.00s label=None caption=there is a man standing in front of a counter with a bunch o...
        [6] t=10.00s label=None caption=someone is holding a cat in a glove while it is sitting on a...
        [7] t=12.00s label=None caption=arafed man standing in a factory with a tablet computer...
        [SEG1] 0.00-4.00s (4.00s) label=work captions=3
            - arafed image of a man working in a factory with a machine
            - someone is working on a laptop with a lot of tools
        [SEG2] 8.00-8.00s (0.00s) label=work captions=1
            - there is a man standing in front of a counter with a bunch of bottles
        [SEG3] 10.00-10.00s (0.00s) label=idle captions=1
            - someone is holding a cat in a glove while it is sitting on a table
        [SEG4] 12.00-12.00s (0.00s) label=work captions=1
            - arafed man standing in a factory with a tablet computer
        [8] t=14.00s label=None caption=arafed worker working on a machine in a factory...
        [9] t=16.00s label=None caption=there is a truck that is parked in a garage...
        [10] t=18.00s label=None caption=there is a computer monitor sitting on a desk in a room...
        [11] t=20.00s label=None caption=arafed worker in a factory working on a machine...
        [SEG5] 14.00-14.00s (0.00s) label=work captions=1
            - arafed worker working on a machine in a factory
        [SEG6] 16.00-18.00s (2.00s) label=idle captions=2
            - there is a truck that is parked in a garage
            - there is a computer monitor sitting on a desk in a room
        [SEG7] 18.00-20.00s (2.00s) label=idle captions=2
            - there is a computer monitor sitting on a desk in a room
            - arafed worker in a factory working on a machine
        [12] t=22.00s label=None caption=arafed worker in a factory working on a machine...
        [13] t=24.00s label=None caption=araffes in a factory with a dog and a man...
        [14] t=26.00s label=None caption=there are people standing in a bus with a dog on the floor...
        [15] t=28.00s label=None caption=arafed man in a factory working on a machine...
        [16] t=30.00s label=None caption=arafed man standing in a factory with a machine behind him...
        [17] t=32.00s label=None caption=arafed man in a factory with a laptop and a box...
        [18] t=34.00s label=None caption=arafed man in a factory holding a large piece of metal...
        [19] t=36.00s label=None caption=there is a man that is standing in a factory with a machine...
        [SEG8] 20.00-22.00s (2.00s) label=work captions=2
            - arafed worker in a factory working on a machine
            - arafed worker in a factory working on a machine
        [SEG9] 32.00-34.00s (2.00s) label=work captions=2
            - arafed man in a factory with a laptop and a box
            - arafed man in a factory holding a large piece of metal
        [SEG10] 34.00-36.00s (2.00s) label=work captions=2
            - arafed man in a factory holding a large piece of metal
            - there is a man that is standing in a factory with a machine
        [20] t=38.00s label=None caption=arafed man standing in a factory with a lot of machines...
        [21] t=40.00s label=None caption=arafed man standing in a factory with a lot of machines...
        [22] t=42.00s label=None caption=arafed man in a blue shirt and mask working on a machine...
        [23] t=44.00s label=None caption=arafed man in a factory looking at machinery in a factory...
        [SEG11] 36.00-38.00s (2.00s) label=work captions=2
            - there is a man that is standing in a factory with a machine
            - arafed man standing in a factory with a lot of machines
        [SEG12] 40.00-44.00s (4.00s) label=work captions=3
            - arafed man standing in a factory with a lot of machines
            - arafed man in a blue shirt and mask working on a machine
        [24] t=46.00s label=None caption=arafed man standing in a factory with a lot of machines...
        [25] t=48.00s label=None caption=arafed man in blue shirt standing in front of a machine...
        [26] t=50.00s label=None caption=arafed man in a factory with a mask on talking on a cell pho...
        [27] t=52.00s label=None caption=arafed man standing in a factory with a lot of machines...
        [SEG13] 42.00-44.00s (2.00s) label=work captions=2
            - arafed man in a blue shirt and mask working on a machine
            - arafed man in a factory looking at machinery in a factory
        [SEG14] 48.00-50.00s (2.00s) label=work captions=2
            - arafed man in blue shirt standing in front of a machine
            - arafed man in a factory with a mask on talking on a cell phone
        [SEG15] 50.00-52.00s (2.00s) label=work captions=2
            - arafed man in a factory with a mask on talking on a cell phone
            - arafed man standing in a factory with a lot of machines
        [28] t=54.00s label=None caption=arafed worker in a factory with a lot of machines...
        [29] t=56.00s label=None caption=arafed worker in a factory with a mask on...
        [30] t=58.00s label=None caption=there is a man that is standing in a store with a plate...
        [31] t=60.00s label=None caption=arafed worker in a factory with boxes and a mask on...
        [SEG16] 52.00-54.00s (2.00s) label=work captions=2
            - arafed man standing in a factory with a lot of machines
            - arafed worker in a factory with a lot of machines
        [SEG17] 56.00-58.00s (2.00s) label=work captions=2
            - arafed worker in a factory with a mask on
            - there is a man that is standing in a store with a plate
        [32] t=62.00s label=None caption=arafed worker in a factory with a mask on and gloves on...
        [33] t=64.00s label=None caption=arafed man in a factory with a machine in the background...
        [SEG18] 60.00-62.00s (2.00s) label=work captions=2
            - arafed worker in a factory with boxes and a mask on
            - arafed worker in a factory with a mask on and gloves on
        [SEG19] 64.00-64.00s (0.00s) label=work captions=1
            - arafed man in a factory with a machine in the background

        ✓ Processing complete!

=== Summary ===
  Total samples: 33
  Total segments: 19

✓ Test completed!
"""
import requests
import json
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Configuration
SERVER_URL = "http://localhost:8001"
VIDEO_PATH = r"C:\Users\BBBS-AI-01\d\behavior_tracking\data\assembly_idle (2) (2).mp4"


def test_blip_llm_4caption_streaming():
    """Test BLIP + LLM with 4-caption windows on a real video."""
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return False
    
    # Step 1: Upload the video
    print("[1] Uploading video...")
    with open(VIDEO_PATH, 'rb') as f:
        files = {'video': f}
        upload_resp = requests.post(f"{SERVER_URL}/backend/upload_vlm", files=files)

    if upload_resp.status_code != 200:
        print(f"Upload failed: {upload_resp.text}")
        return False

    upload_data = upload_resp.json()
    filename = upload_data.get('filename')
    print(f"✓ Uploaded as: {filename}")

    # Step 2: Stream processing with VLM + LLM (4-caption windows)
    print("\n[2] Processing video with BLIP + LLM (4-caption windows)...")
    stream_url = (
        f"{SERVER_URL}/backend/vlm_local_stream"
        f"?filename={filename}"
        f"&model=Salesforce/blip-image-captioning-large"
        f"&classifier_source=llm"
        f"&classifier_mode=binary"
        f"&sample_interval_sec=2.0"
    )

    segment_count = 0
    sample_count = 0
    analysis_id = None

    try:
        with requests.get(stream_url, stream=True, timeout=300) as resp:
            if resp.status_code != 200:
                print(f"Stream failed: {resp.status_code} {resp.text}")
                return False
            
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                if not line_str.startswith('data: '):
                    continue
                
                try:
                    payload = json.loads(line_str[6:])  # strip 'data: '
                    stage = payload.get('stage', '')
                    
                    if stage == 'video_info':
                        fps = payload.get('fps')
                        frame_count = payload.get('frame_count')
                        duration = payload.get('duration')
                        print(f"  Video: {frame_count} frames @ {fps} fps, {duration:.1f}s duration")
                    
                    elif stage == 'sample':
                        sample_count += 1
                        t = payload.get('time_sec')
                        caption = payload.get('caption', '').replace('\n', ' ')[:60]
                        label = payload.get('label')
                        print(f"  [{sample_count}] t={t:.2f}s label={label} caption={caption}...")
                    
                    elif stage == 'segment':
                        segment_count += 1
                        start = payload.get('start_time')
                        end = payload.get('end_time')
                        label = payload.get('label')
                        duration = payload.get('duration')
                        captions = payload.get('captions', [])
                        print(f"  [SEG{segment_count}] {start:.2f}-{end:.2f}s ({duration:.2f}s) label={label} captions={len(captions)}")
                        for cap in captions[:2]:
                            print(f"       - {cap[:70]}")
                    
                    elif stage == 'finished':
                        print(f"\n✓ Processing complete!")
                        analysis_id = payload.get('stored_analysis_id')
                        if analysis_id:
                            print(f"  Stored analysis ID: {analysis_id}")
                        print(f"\n=== Summary ===")
                        print(f"  Total samples: {sample_count}")
                        print(f"  Total segments: {segment_count}")
                    
                    elif stage == 'alert':
                        print(f"  ⚠ {payload.get('message')}")
                    
                    elif stage == 'debug':
                        print(f"  [DEBUG] {payload.get('message')}")
                
                except json.JSONDecodeError:
                    pass

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False

    print("\n✓ Test completed!")
    return analysis_id is not None


if __name__ == '__main__':
    success = test_blip_llm_4caption_streaming()
    sys.exit(0 if success else 1)

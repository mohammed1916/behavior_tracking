"""Quick SSE segment test - assumes video is already uploaded."""
import urllib.request
import json
import time

def test_segment_events():
    """Quick test - manually upload video first, then run this."""
    
    # Assumes you manually uploaded assembly_drone_240_144.mp4 via frontend first
    # Or use curl: curl -F "video=@data/assembly_drone/assembly_drone_240_144.mp4" http://localhost:8001/backend/upload_vlm
    
    print("\n" + "="*60)
    print("QUICK SSE SEGMENT EVENT TEST")
    print("="*60)
    
    # First, upload the video
    import os
    video_path = r"C:\Users\BBBS-AI-01\d\behavior_tracking\data\assembly_drone\assembly_drone_240_144.mp4"
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return
        
    print(f"\n1. Uploading video: {os.path.basename(video_path)}")
    
    # Prepare multipart upload
    import io
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = io.BytesIO()
    
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    body.write(f'--{boundary}\r\n'.encode())
    body.write(b'Content-Disposition: form-data; name="video"; filename="assembly_drone_240_144.mp4"\r\n')
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
            upload_result = json.loads(resp.read().decode())
            filename = upload_result['filename']
            print(f"   ✓ Uploaded as: {filename}")
    except Exception as e:
        print(f"   ✗ Upload failed: {e}")
        return
    
    # Now stream with LLM mode
    print(f"\n2. Streaming with classifier_source=llm")
    url = (
        f"http://localhost:8001/backend/vlm_local_stream?"
        f"filename={filename}"
        f"&model=qwen/qwen2-vl-2b-instruct"
        f"&prompt=Describe+activity"
        f"&classifier_source=llm"
        f"&classifier_mode=multi"
        f"&processing_mode=every_2s"
    )
    
    print(f"   URL: {url[:80]}...")
    
    events = []
    segments = []
    samples = []
    
    try:
        req = urllib.request.Request(url)
        print(f"   Opening SSE stream...")
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                if line_str.startswith('data: '):
                    try:
                        data = json.loads(line_str[6:])
                        events.append(data)
                        stage = data.get('stage', '')
                        
                        if stage == 'started':
                            print(f"   ✓ Stream started")
                        elif stage == 'video_info':
                            fps = data.get('fps', 0)
                            frames = data.get('frame_count', 0)
                            print(f"   ✓ Video info: {frames} frames @ {fps:.1f} fps")
                        elif stage == 'sample':
                            samples.append(data)
                            if len(samples) == 1:
                                print(f"   ✓ First sample received")
                        elif stage == 'segment':
                            segments.append(data)
                            seg_label = data.get('label', 'unknown')
                            seg_dur = data.get('duration', 0)
                            print(f"   ✓ Segment #{len(segments)}: {seg_label} ({seg_dur:.2f}s)")
                        elif stage == 'finished':
                            print(f"   ✓ Stream finished")
                            break
                    except json.JSONDecodeError as e:
                        print(f"   ! JSON parse error: {e}")
    except Exception as e:
        print(f"   ✗ Stream error: {e}")
    
    # Report results
    print(f"\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total events:   {len(events)}")
    print(f"Sample events:  {len(samples)}")
    print(f"Segment events: {len(segments)}")
    
    if segments:
        print(f"\n✓ SUCCESS: Segments emitted in LLM mode")
        print(f"\nSegment details:")
        for i, seg in enumerate(segments):
            print(f"  {i+1}. {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s: "
                  f"{seg.get('label', 'unknown')} "
                  f"(dominant: {seg.get('dominant_caption', '')[:50]}...)")
            if 'llm_output' in seg:
                print(f"     LLM output: {seg.get('llm_output', '')[:80]}")
    else:
        print(f"\n✗ FAIL: No segment events received (expected > 0 for LLM mode)")
    
    # Verify required fields in first segment
    if segments:
        seg = segments[0]
        required = ['stage', 'start_time', 'end_time', 'duration', 'dominant_caption', 'timeline', 'label']
        missing = [f for f in required if f not in seg]
        if missing:
            print(f"\n✗ FAIL: Missing fields in segment: {missing}")
        else:
            print(f"\n✓ All required segment fields present")
    
    print("="*60)

if __name__ == '__main__':
    test_segment_events()

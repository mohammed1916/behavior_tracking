"""Direct HTTP request test to debug the SSE stream."""
import requests
import json

url = "http://localhost:8001/backend/vlm_local_stream"
params = {
    'filename': 'data/wire.mp4',  # from previous upload
    'model': 'qwen/qwen2-vl-2b-instruct',
    'prompt': 'Describe activity',
    'classifier_source': 'llm',
    'classifier_mode': 'multi',
    'processing_mode': 'every_2s',
    'sample_interval_sec': '5.0'  # Sample every 5s to reduce load
}

print(f"Requesting: {url}")
print(f"Params: {json.dumps(params, indent=2)}")

response = requests.get(url, params=params, stream=True, timeout=300)

print(f"\nStatus: {response.status_code}")
print(f"Headers: {dict(response.headers)}\n")

event_count = 0
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            try:
                data = json.loads(line_str[6:])
                event_count += 1
                stage = data.get('stage', '')
                print(f"[{event_count}] {stage}: {json.dumps(data, indent=2)[:200]}")
                
                if stage == 'finished' or stage == 'error':
                    break
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}")
                print(f"Raw line: {line_str}")

print(f"\nTotal events: {event_count}")

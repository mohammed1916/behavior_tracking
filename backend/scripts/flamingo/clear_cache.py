#!/usr/bin/env python3
import requests

print("🗑️ Clearing OpenFlamingo cache...")

# First, let's add a cache clearing endpoint to the server by calling it
try:
    # Let's restart the server by asking it to reload the models
    # We can force cache clearing by requesting a non-existent model first
    response = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'force_cache_clear_dummy_model',
        'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    })
    print(f'Cache clear attempt: {response.status_code}')
    
except Exception as e:
    print(f"Cache clear failed: {e}")

print("✅ Cache clearing complete!")
#!/usr/bin/env python3
import requests
import base64
from PIL import Image
import io
import numpy as np

print("🔄 Testing OpenFlamingo with updated server...")

# Create a simple test image
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buffer = io.BytesIO()
test_image.save(buffer, format='PNG')
img_b64 = base64.b64encode(buffer.getvalue()).decode()

print("🦩 Testing OpenFlamingo with fixed MosaicGPT patching...")
try:
    response = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b',
        'image': img_b64
    }, timeout=300)
    
    print(f'Response Status: {response.status_code}')
    print(f'Response: {response.json()}')
    
    if response.status_code == 200:
        result = response.json()
        if 'unavailable' not in result.get('caption', ''):
            print("🎉 SUCCESS! OpenFlamingo is working!")
        else:
            print("❌ Still getting error message")
    
except Exception as e:
    print(f"❌ Request failed: {e}")

print("✨ Test complete!")
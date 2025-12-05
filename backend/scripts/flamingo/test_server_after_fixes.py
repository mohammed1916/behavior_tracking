#!/usr/bin/env python3
"""
Clear OpenFlamingo cache and test with complete fixes
"""
import requests

print("🗑️ Forcing OpenFlamingo cache clear and testing...")

# Force cache clearing by making a request that will clear the cache
try:
    print("1️⃣ Clearing cache by requesting unknown model...")
    clear_response = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'clear_flamingo_cache_dummy',
        'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    })
    print(f"   Cache clear response: {clear_response.status_code}")
    
    print("2️⃣ Testing OpenFlamingo with updated server code...")
    test_response = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b',
        'image': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    }, timeout=300)
    
    result = test_response.json()
    print(f"✅ OpenFlamingo test result:")
    print(f"   Status: {test_response.status_code}")
    print(f"   Caption: {result.get('caption', 'N/A')}")
    print(f"   Model: {result.get('model', 'N/A')}")
    
    if 'unavailable' not in result.get('caption', ''):
        print("🎉 SUCCESS! OpenFlamingo is working!")
    else:
        print("⚠️ Still getting cached error message")
        
except Exception as e:
    print(f"❌ Test failed: {e}")

print("✨ Test complete!")
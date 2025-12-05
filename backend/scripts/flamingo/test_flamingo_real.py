#!/usr/bin/env python3
"""Test OpenFlamingo with real captured image"""
import base64
import requests
import os

def test_flamingo():
    # Use one of the captured frames
    image_path = 'captured_frames/idle_1764671568.jpg'
    
    if os.path.exists(image_path):
        print(f'Testing with real captured image: {image_path}')
        
        # Load and convert to base64
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_b64 = base64.b64encode(img_data).decode()
        
        print('Testing OpenFlamingo...')
        try:
            response_flamingo = requests.post('http://localhost:8001/backend/caption', json={
                'model': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b',
                'image': img_b64
            }, timeout=300)
            
            result = response_flamingo.json()
            print(f'OpenFlamingo Response ({response_flamingo.status_code}):')
            print(f'   Caption: {result.get("caption", "N/A")}')
            print(f'   Model: {result.get("model", "N/A")}')
            print(f'   Status: {result.get("status", "N/A")}')
            
            if 'unavailable' not in result.get('caption', ''):
                print('SUCCESS! OpenFlamingo is working!')
                return True
            else:
                print('OpenFlamingo returned error message')
                return False
                
        except requests.exceptions.Timeout:
            print('Request timed out - model may be loading')
            return False
        except Exception as e:
            print(f'Request failed: {e}')
            return False
        
    else:
        print(f'Image not found: {image_path}')
        return False

def compare_with_blip():
    image_path = 'captured_frames/idle_1764671568.jpg'
    
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_b64 = base64.b64encode(img_data).decode()
        
        print('\nComparing with BLIP...')
        response_blip = requests.post('http://localhost:8001/backend/caption', json={
            'model': 'Salesforce/blip-image-captioning-large',
            'image': img_b64
        })
        
        blip_result = response_blip.json()
        print(f'BLIP Caption: {blip_result.get("caption", "N/A")}')

if __name__ == "__main__":
    flamingo_works = test_flamingo()
    compare_with_blip()
    
    if flamingo_works:
        print("\n✅ Test complete - OpenFlamingo is working!")
    else:
        print("\n❌ Test complete - OpenFlamingo needs more work")
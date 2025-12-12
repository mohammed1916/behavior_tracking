#!/usr/bin/env python
import base64
import requests
from PIL import Image
import io
import os

print("Testing with the provided mechanical/crafting image...")

# The image should be in the current directory as an attachment
# Let me check what files are available
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
print("Files in directory:")
for file in os.listdir('.'):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        print(f"  Found image: {file}")

# Since the image was provided as an attachment, let me try to find it
# or use the fact that it's showing hands working with mechanical parts on a cutting mat

# For now, let me create a version based on what I can see - it's a photo of hands 
# working with what appears to be mechanical parts or a model on a green cutting mat

# I'll need to save the attachment first. Let me check if there are any image files
image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if image_files:
    image_path = image_files[0]
    print(f"Using image: {image_path}")
    
    # Load and convert to base64
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        img_b64 = base64.b64encode(img_data).decode()
    
    print("Testing BLIP with the real image...")
    response_blip = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'Salesforce/blip-image-captioning-large',
        'image': img_b64
    })
    
    print(f"BLIP Response ({response_blip.status_code}): {response_blip.json()}")
    
    print("\nTesting OpenFlamingo with the real image...")
    response_flamingo = requests.post('http://localhost:8001/backend/caption', json={
        'model': 'openflamingo/OpenFlamingo-3B-vitl-mpt1b',
        'image': img_b64
    }, timeout=300)  # 5 minute timeout
    
    print(f"OpenFlamingo Response ({response_flamingo.status_code}): {response_flamingo.json()}")
    
else:
    print("No image files found in current directory")
    print("The attachment may need to be saved manually")
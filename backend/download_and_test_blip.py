import sys
import traceback

print('Starting BLIP download/load test...')
try:
    from transformers import pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print('Transformers and torch available. device=', device)
    # This will download model weights if not cached
    p = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large', device=device)
    print('Pipeline created successfully.')
    # Run a dummy call with a tiny blank image to ensure weights are loaded into memory
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.zeros((64,64,3), dtype='uint8')))
    # ImageToTextPipeline may not accept `max_length`; call without that arg
    out = p(img)
    print('Model responded:', out)
    print('BLIP load test completed successfully.')
except Exception as e:
    print('Error during BLIP load test:')
    traceback.print_exc()
    sys.exit(1)

# python -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/blip-image-captioning-large')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download('InternRobotics/G2VLM-2B-MoT')" # 18.2 GB
# python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')"
# python -c "from huggingface_hub import snapshot_download; snapshot_download('omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321')"
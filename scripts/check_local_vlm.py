import json
import sys
res = {
    'transformers': None,
    'torch': None,
    'cuda': False,
    'pipeline_load': False,
    'error': None,
}
try:
    import transformers
    res['transformers'] = getattr(transformers, '__version__', str(transformers))
except Exception as e:
    res['error'] = f'transformers import error: {e}'
    print(json.dumps(res))
    sys.exit(0)
try:
    import torch
    res['torch'] = getattr(torch, '__version__', str(torch))
    res['cuda'] = torch.cuda.is_available()
except Exception as e:
    res['error'] = f'torch import error: {e}'
    print(json.dumps(res))
    sys.exit(0)
from transformers import pipeline
try:
    device = 0 if res['cuda'] else -1
    # local_files_only=True prevents downloads; will raise if model not cached
    p = pipeline('image-to-text', model='Salesforce/blip-image-captioning-large', device=device, local_files_only=True)
    res['pipeline_load'] = True
except Exception as e:
    res['error'] = str(e)
print(json.dumps(res))

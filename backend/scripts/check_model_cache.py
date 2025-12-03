import os
import json
import sys
import pkgutil

res = {
    'transformers_installed': pkgutil.find_loader('transformers') is not None,
    'torch_installed': pkgutil.find_loader('torch') is not None,
    'hf_cache_paths': [],
    'model_present': False,
}

# Common HF cache locations
home = os.path.expanduser('~')
possible = [
    os.path.join(home, '.cache', 'huggingface', 'transformers'),
    os.path.join(home, '.cache', 'huggingface', 'hub'),
    os.path.join(home, '.cache', 'huggingface'),
    os.path.join(home, '.huggingface', 'transformers'),
]

for p in possible:
    if os.path.exists(p):
        res['hf_cache_paths'].append(p)

# If cache dirs found, search for model name
model_name = 'blip-image-captioning-large'
for base in res['hf_cache_paths']:
    for root, dirs, files in os.walk(base):
        if model_name in root or any(model_name in d for d in dirs):
            res['model_present'] = True
            res['model_path'] = root
            break
    if res['model_present']:
        break

print(json.dumps(res))

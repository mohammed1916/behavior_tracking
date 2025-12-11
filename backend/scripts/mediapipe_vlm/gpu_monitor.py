import time
from datetime import datetime

try:
    import torch
except Exception:
    torch = None

import pandas as pd

import os
from config import GPU_CSV_FILE 

def get_gpu_memory_info():
    """Return GPU memory usage information in MB.

    Returns a dict with keys: allocated_mb, reserved_mb, max_allocated_mb, device_index
    If CUDA is not available, returns zeros.
    """
    if torch is None:
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'max_allocated_mb': 0.0,
            'device_index': None,
            'cuda_available': False
        }

    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'max_allocated_mb': 0.0,
            'device_index': None,
            'cuda_available': False
        }

    dev = torch.cuda.current_device()
    allocated = float(torch.cuda.memory_allocated(dev)) / (1024 ** 2)
    reserved = float(torch.cuda.memory_reserved(dev)) / (1024 ** 2)
    max_alloc = float(torch.cuda.max_memory_allocated(dev)) / (1024 ** 2)

    return {
        'allocated_mb': round(allocated, 2),
        'reserved_mb': round(reserved, 2),
        'max_allocated_mb': round(max_alloc, 2),
        'device_index': int(dev),
        'cuda_available': True
    }


def log_gpu_usage(label=None, csv_file=GPU_CSV_FILE):
    """Append current GPU usage to a CSV file with timestamp and optional label."""
    info = get_gpu_memory_info()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        'timestamp': timestamp,
        'label': label or '',
        'cuda_available': info['cuda_available'],
        'device_index': info['device_index'],
        'allocated_mb': info['allocated_mb'],
        'reserved_mb': info['reserved_mb'],
        'max_allocated_mb': info['max_allocated_mb']
    }

    try:
        df = pd.DataFrame([row])
        # Append to CSV (create if not exists)
        if not pd.io.common.file_exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)
    except Exception:
        # Best-effort: if pandas not available or write fails, print to stdout
        print(f"GPU: {timestamp} | {row}")

    return row

import os
import sys
import time
from PIL import Image

here = os.path.dirname(__file__)
MODEL_DIR = os.path.join(here, 'qwen_vlm_2b_activity_model')

def main():
    print('Qwen model dir:', MODEL_DIR)
    # Force CPU to avoid GPU OOM during quick local tests. Set to empty to disable CUDA visibility.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    try:
        import importlib.util
        vlm_path = os.path.join(os.path.dirname(__file__), '..', 'vlm_qwen.py')
        vlm_path = os.path.abspath(vlm_path)
        spec = importlib.util.spec_from_file_location('vlm_qwen', vlm_path)
        vlm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vlm_mod)
        QwenVLMAdapter = getattr(vlm_mod, 'QwenVLMAdapter')
    except Exception as e:
        print('Failed to import QwenVLMAdapter by path:', e)
        sys.exit(1)

    if not os.path.exists(MODEL_DIR):
        print('Model dir not found:', MODEL_DIR)
        sys.exit(1)

    print('Instantiating adapter (this may download or load large weights; CPU may be slow)...')
    t0 = time.time()
    try:
        adapter = QwenVLMAdapter(MODEL_DIR)
    except Exception as e:
        print('Adapter init failed:', e)
        sys.exit(1)
    print('Adapter ready, elapsed %.1fs' % (time.time() - t0))

    # build a small test image (red square)
    img = Image.new('RGB', (224, 224), (255, 0, 0))
    print('Running inference...')
    out = adapter(img)
    print('Raw adapter output:', out)

    try:
        adapter.release()
    except Exception:
        pass

if __name__ == '__main__':
    main()

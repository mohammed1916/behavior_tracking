from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import uuid
from tracker import BehaviorTracker
from typing import Optional, Callable
from fastapi import Query
import json
import logging
import subprocess

# configure logging so INFO/DEBUG messages are shown
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# write logs to a file so the frontend can tail recent activity
LOG_FILE = "server.log"
logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(LOG_FILE) for h in logger.handlers):
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)

# Local VLM (BLIP) imports will be optional and loaded lazily
_local_captioner = None

# Prompt template for LLM classification of captions (work vs idle).
# Use .format(prompt=..., caption=...) to substitute values.
CLASSIFY_PROMPT_TEMPLATE = (
    "Decide whether the following caption describes active 'work' or 'idle' (single word response: work or idle).\n\n"
    "Context prompt: {prompt}\nCaption: {caption}\n"
)

# Cache captioners by model id so repeated requests don't reload them
_captioner_cache = {}

def get_local_captioner():
    """Return a default local captioner (BLIP) if available."""
    global _local_captioner
    if _local_captioner is not None:
        return _local_captioner
    try:
        from transformers import pipeline
        import torch
    except Exception:
        _local_captioner = None
        return None

    device = 0 if torch.cuda.is_available() else -1
    try:
        _local_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device, local_files_only=True)
        logging.info('Loaded default local captioner Salesforce/blip-image-captioning-large')
    except Exception:
        _local_captioner = None
    return _local_captioner


def get_captioner_for_model(model_id: str):
    """Return a captioner pipeline for `model_id`.
    Tries local cache first (`local_files_only=True`), then falls back to downloading the model.
    """
    global _captioner_cache
    if not model_id:
        return get_local_captioner()

    if model_id in _captioner_cache:
        return _captioner_cache[model_id]

    try:
        from transformers import pipeline
        import torch
    except Exception:
        _captioner_cache[model_id] = None
        return None

    # Choose device: prefer GPU if available and it has sufficient free memory,
    # otherwise fall back to CPU to avoid CUDA OOM during model load.
    device = 0 if torch.cuda.is_available() else -1
    try:
        if device >= 0:
            # torch.cuda.mem_get_info(device) -> (free, total)
            try:
                free, total = torch.cuda.mem_get_info(device)
            except Exception:
                free = None
                total = None
            # Require a safety margin (e.g., 2 GiB) to attempt GPU loads
            MIN_GPU_FREE = 2 * 1024 * 1024 * 1024
            if free is not None and free < MIN_GPU_FREE:
                logging.info('GPU available but free memory (%s bytes) < %s bytes; preferring CPU to avoid OOM', free, MIN_GPU_FREE)
                device = -1
    except Exception:
        # If querying GPU memory fails for any reason, silently prefer CPU fallback to be safe
        device = -1

    # Default task; try to read from local_vlm_models.json if present
    task = 'image-to-text'
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            for e in cfg.get('models', []):
                if e.get('id') == model_id:
                    task = e.get('task', task)
                    break
    except Exception:
        pass

    # Try local-only load first
    try:
        p = pipeline(task, model=model_id, device=device, local_files_only=True)
        _captioner_cache[model_id] = p
        logging.info('Loaded captioner for model %s (task=%s) from local cache on device=%s', model_id, task, device)
        return p
    except Exception as ex_local:
        logging.info('Local-only load failed for %s: %s', model_id, ex_local)
        # If this looks like a CUDA OOM / CUDA error, try falling back to CPU to avoid OOM
        ex_text = str(ex_local).lower()
        tried_cpu_fallback = False
        try:
            if 'out of memory' in ex_text or 'cuda' in ex_text or 'cudart' in ex_text:
                logging.info('Detected CUDA-related error while loading %s, attempting CPU fallback...', model_id)
                try:
                    # free some GPU memory if possible
                    try:
                        import torch as _torch_for_cleanup
                        if _torch_for_cleanup.cuda.is_available():
                            _torch_for_cleanup.cuda.empty_cache()
                    except Exception:
                        pass
                    p = pipeline(task, model=model_id, device=-1, local_files_only=True)
                    _captioner_cache[model_id] = p
                    logging.info('Loaded captioner for model %s on CPU (fallback)', model_id)
                    return p
                except Exception as ex_cpu:
                    logging.info('CPU fallback failed for %s: %s', model_id, ex_cpu)
                tried_cpu_fallback = True
        except Exception:
            tried_cpu_fallback = False

        # Attempt a fallback download (may be slow). If the earlier failure was CUDA-related
        # and we haven't yet tried CPU, try downloading then falling back to CPU on failure.
        try:
            logging.info('Attempting to download model %s (task=%s)...', model_id, task)
            p = pipeline(task, model=model_id, device=device)
            _captioner_cache[model_id] = p
            logging.info('Downloaded and loaded captioner for model %s (task=%s)', model_id, task)
            return p
        except Exception as ex_fetch:
            logging.info('Failed to download/load captioner for %s on device=%s: %s', model_id, device, ex_fetch)
            # If download/load failed and we haven't yet tried CPU, try CPU download/load (may still OOM but often works)
            if not tried_cpu_fallback:
                try:
                    logging.info('Attempting to download/load %s on CPU...', model_id)
                    p = pipeline(task, model=model_id, device=-1)
                    _captioner_cache[model_id] = p
                    logging.info('Downloaded and loaded captioner for %s on CPU', model_id)
                    return p
                except Exception as ex_cpu_fetch:
                    logging.info('Failed to download/load captioner for %s on CPU: %s', model_id, ex_cpu_fetch)

            _captioner_cache[model_id] = None
            return None


# Lazy loader for a local text LLM (optional)
_local_text_llm = None
def get_local_text_llm():
    global _local_text_llm
    if _local_text_llm is not None:
        return _local_text_llm

    # Prefer Ollama CLI if available
    try:
        if shutil.which('ollama'):
            ollama_model = os.environ.get('OLLAMA_MODEL', 'strat')
            class OllamaWrapper:
                def __init__(self, model_id):
                    self.model_id = model_id
                def __call__(self, prompt, max_new_tokens=128):
                    try:
                        # Ensure subprocess output is decoded using UTF-8 and
                        # replace undecodable bytes to avoid UnicodeDecodeError
                        p = subprocess.run(
                            ['ollama', 'run', self.model_id, prompt],
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            timeout=120,
                        )
                        out = (p.stdout or '').strip()
                        return [{'generated_text': out}]
                    except Exception as e:
                        logging.info('Ollama run failed: %s', e)
                        return [{'generated_text': ''}]
            _local_text_llm = OllamaWrapper(ollama_model)
            logging.info('Using Ollama CLI model %s for text LLM', ollama_model)
            return _local_text_llm
    except Exception:
        pass

    try:
        from transformers import pipeline
        import torch
    except Exception:
        _local_text_llm = None
        return None
    device = 0 if torch.cuda.is_available() else -1
    # Try a small local-capable HF model first (must be cached locally to avoid downloads)
    candidates = [
        'Qwen/Qwen-2.5',
        'qwen/Qwen-2.5',
        'gpt2',
    ]
    for cid in candidates:
        try:
            _local_text_llm = pipeline('text-generation', model=cid, device=device, local_files_only=True)
            return _local_text_llm
        except Exception:
            _local_text_llm = None
    return None


def _normalize_caption_output(captioner, out):
    """Normalize various pipeline outputs into a text string.
    Some models/pipelines may return dicts containing token ids (e.g. 'input_ids').
    Try to decode those using the pipeline's tokenizer when possible.
    """
    try:
        first = None
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
        else:
            first = out

        if isinstance(first, dict):
            # common keys
            for k in ('generated_text', 'caption', 'text'):
                if k in first and first.get(k):
                    return first.get(k)

            # If token ids are present, try to decode via tokenizer
            if 'input_ids' in first:
                ids = first.get('input_ids')
                try:
                    # handle tensors or nested lists
                    if hasattr(ids, 'tolist'):
                        ids_list = ids.tolist()
                    else:
                        ids_list = list(ids)
                    # flatten if necessary
                    if isinstance(ids_list, (list,)) and len(ids_list) > 0 and isinstance(ids_list[0], (list,)):
                        ids_list = ids_list[0]
                except Exception:
                    ids_list = None

                if ids_list and hasattr(captioner, 'tokenizer'):
                    try:
                        return captioner.tokenizer.decode(ids_list, skip_special_tokens=True)
                    except Exception:
                        pass

            return str(first)

        return str(first)
    except Exception:
        try:
            return str(out)
        except Exception:
            return ''

def process_video_samples(file_path: str, captioner=None, max_samples: int = 30, stream_callback: Optional[Callable] = None, prompt: str = '', use_llm: bool = False):
    """Process a video file: compute basic video_info and sample frames to caption.

    Args:
        file_path (str): Path to the video file.
        captioner: Optional captioning model to use for frame captions.
        max_samples (int): Maximum number of frames to sample and caption.
        stream_callback (Optional[Callable]): If provided, called with event dicts for each stage/sample.
        prompt (str): Optional prompt to guide the captioning or LLM, if applicable.
        use_llm (bool): If True, use a language model (LLM) for captioning or analysis.

    Returns:
        dict: A dictionary with video_info, samples, idle_frames, work_frames.
    """
    result = {}
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            msg = {"stage": "error", "message": "could not open video"}
            if stream_callback:
                stream_callback(msg)
            else:
                result.update({"error": "could not open video"})
            return result

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / (fps if fps > 0 else 30.0)

        video_info = {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}
        if stream_callback:
            stream_callback({"stage": "video_info", **video_info})

        total_frames = max(1, frame_count)
        desired = min(max_samples, max(1, total_frames))
        step = max(1, total_frames // desired) if total_frames > 0 else 1

        samples = []
        work_frames = []
        idle_frames = []

        # keyword classifier (backup)
        work_keywords = [
            'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
            'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
            'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
        ]

        # Optionally prepare a local text LLM for semantic classification
        text_llm = None
        if use_llm:
            try:
                text_llm = get_local_text_llm()
                if text_llm is None:
                    logging.info('LLM classification requested but no local text LLM available')
            except Exception:
                text_llm = None

        if captioner is None:
            captioner = get_local_captioner()

        if captioner is None:
            msg = {"stage": "error", "message": "no local captioner available"}
            if stream_callback:
                stream_callback(msg)
            else:
                result.update({"local_captioner": "not available"})
            cap.release()
            return result

        cap2 = cv2.VideoCapture(file_path)
        if not cap2.isOpened():
            msg = {"stage": "error", "message": "could not open video for sampling"}
            if stream_callback:
                stream_callback(msg)
            else:
                result.update({"error": "could not open video for sampling"})
            cap.release()
            return result

        for idx in range(0, int(total_frames), step):
            if len(samples) >= max_samples:
                break
            try:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, f = cap2.read()
                if not ret or f is None:
                    if stream_callback:
                        stream_callback({"stage": "sample_error", "frame_index": idx, "error": "read_failed"})
                    else:
                        samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": "read_failed"})
                    continue

                # caption
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil = Image.fromarray(rgb)
                try:
                    out = captioner(pil)
                except Exception as ex:
                    if stream_callback:
                        stream_callback({"stage": "sample_error", "frame_index": idx, "error": str(ex)})
                    else:
                        samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": str(ex)})
                        idle_frames.append(idx)
                    continue

                text = _normalize_caption_output(captioner, out)

                # Decide label: prefer LLM-based classification if requested and available,
                # otherwise fall back to simple keyword matching.
                is_work = False
                if use_llm and text_llm is not None:
                    try:
                        cls_prompt = CLASSIFY_PROMPT_TEMPLATE.format(prompt=prompt, caption=text)
                        out = text_llm(cls_prompt, max_new_tokens=8)
                        out_text = None
                        if isinstance(out, list) and len(out) > 0:
                            first = out[0]
                            if isinstance(first, dict):
                                out_text = first.get('generated_text') or first.get('text') or str(first)
                            else:
                                out_text = str(first)
                        else:
                            out_text = str(out)
                        if out_text:
                            out_text = out_text.lower()
                            if 'work' in out_text and 'idle' not in out_text:
                                is_work = True
                            elif 'idle' in out_text and 'work' not in out_text:
                                is_work = False
                            else:
                                # ambiguous -> fallback to keyword
                                is_work = any(k in (text or '').lower() for k in work_keywords)
                    except Exception as e:
                        logging.info('LLM classifier failed: %s', e)
                        is_work = any(k in (text or '').lower() for k in work_keywords)
                else:
                    text_lower = (text or '').lower()
                    is_work = any(k in text_lower for k in work_keywords)

                sample = {"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "caption": text, "label": "work" if is_work else "idle"}
                if stream_callback:
                    stream_callback({"stage": "sample", **sample})
                else:
                    samples.append(sample)
                    if is_work:
                        work_frames.append(idx)
                    else:
                        idle_frames.append(idx)
            except Exception as e:
                if stream_callback:
                    stream_callback({"stage": "sample_error", "frame_index": idx, "error": str(e)})
                else:
                    samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": str(e)})

        cap2.release()
        cap.release()

        if not stream_callback:
            result.update({"video_info": video_info, "samples": samples, "idle_frames": idle_frames, "work_frames": work_frames})
        return result
    except Exception as e:
        if stream_callback:
            stream_callback({"stage": "error", "message": str(e)})
            return {}
        return {"error": str(e)}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
VLM_UPLOAD_DIR = "vlm_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VLM_UPLOAD_DIR, exist_ok=True)

@app.post("/backend/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Process video
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = BehaviorTracker()
    task_completed = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, state, completed = tracker.process_frame(frame)
        if completed:
            task_completed = True
            
        out.write(processed_frame)
        
    cap.release()
    out.release()
    
    return {
        "message": "Video processed successfully",
        "task_completed": task_completed,
        "processed_video_path": output_path,
        "download_url": f"/backend/download/{output_filename}"
    }

@app.get("/backend/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename, headers={"Accept-Ranges": "bytes"})
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/backend/stream_pose")
async def stream_pose():
    """Stream webcam frames with only pose/hand landmarks."""
    def gen_frames():
        cap = cv2.VideoCapture(0)
        tracker = BehaviorTracker()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Mirror frame for selfie view so landmarks match user's expectation
            frame = cv2.flip(frame, 1)
            processed_frame, _, _ = tracker.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post("/backend/vlm")
async def vlm_endpoint(
    model: str = Form(...),
    prompt: str = Form(''),
    video: UploadFile = File(None),
    use_llm: bool = Form(False),
):
    """VLM endpoint stub for videos: saves optional video and returns basic analysis.
    Placeholder — extend to call a real VLM when available.
    """
    video_url = None
    analysis = {"model": model, "prompt": prompt}

    if video:
        # Save uploaded video
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{video.filename}"
        save_path = os.path.join(VLM_UPLOAD_DIR, filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # duration, fps, width, height
        try:
            cap = cv2.VideoCapture(save_path)
            if not cap.isOpened():
                analysis.update({"warning": "could not open saved video"})
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                duration = frame_count / fps if fps > 0 else 0

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    avg_color_per_row = frame.mean(axis=0).mean(axis=0)
                    avg_color = [float(avg_color_per_row[2]), float(avg_color_per_row[1]), float(avg_color_per_row[0])]
                    analysis.update({
                        "width": width,
                        "height": height,
                        "fps": float(fps),
                        "frame_count": int(frame_count),
                        "duration_sec": float(duration),
                        "first_frame_avg_color_bgr": avg_color,
                    })
                else:
                    analysis.update({"warning": "could not read first frame"})
                cap.release()
        except Exception as e:
            analysis.update({"error": str(e)})

        video_url = f"/backend/vlm_video/{filename}"
        analysis["video_url"] = video_url

    # If a local captioner is available for the requested model, run the sampling/captioning helper
    captioner = get_captioner_for_model(model)
    if captioner is not None:
        try:
            proc = process_video_samples(save_path, captioner=captioner, max_samples=30, stream_callback=None, prompt=prompt, use_llm=use_llm)
            # merge proc results into analysis
            if isinstance(proc, dict):
                # if video_info present, expose fields at top-level for backward compat
                vi = proc.get('video_info') or {}
                analysis.update(vi)
                if 'samples' in proc:
                    analysis['samples'] = proc.get('samples')
                if 'idle_frames' in proc:
                    analysis['idle_frames'] = proc.get('idle_frames')
                if 'work_frames' in proc:
                    analysis['work_frames'] = proc.get('work_frames')
            analysis["video_url"] = video_url
            return {"message": "VLM (video) local response", "analysis": analysis}
        except Exception as e:
            logging.exception('Local VLM processing failed')
            analysis.update({"error": str(e)})

    # Fallback/stub response (no local captioner available or processing failed)
    return {"message": "VLM (video) stub response", "analysis": analysis}


@app.post("/backend/upload_vlm")
async def upload_vlm(video: UploadFile = File(...)):
    """Save an uploaded video to `vlm_uploads/` and return the saved filename.
    This is used when the frontend wants to start a streaming processing session.
    """
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{video.filename}"
    save_path = os.path.join(VLM_UPLOAD_DIR, filename)
    with open(save_path, "wb") as buffer:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    logging.info(f"Saved VLM upload to {save_path}")
    return {"filename": filename}


def _sse_event(data: dict, event: Optional[str] = None) -> bytes:
    payload = ''
    if event:
        payload += f"event: {event}\n"
    payload += "data: " + json.dumps(data, default=str) + "\n\n"
    return payload.encode('utf-8')


@app.get("/backend/vlm_local_stream")
async def vlm_local_stream(filename: str = Query(...), model: str = Query(...), prompt: str = Query(''), use_llm: bool = Query(False)):
    """Stream processing events (SSE) for a previously-uploaded VLM video.
    The frontend should first POST the file to `/backend/upload_vlm` and then open
    an EventSource to this endpoint with the returned `filename`.
    """
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    def gen():
        try:
            yield _sse_event({"stage": "started", "message": "processing started"})

            # Basic video info
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                yield _sse_event({"stage": "error", "message": "failed to open video"})
                return
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / (fps if fps > 0 else 30.0)
            yield _sse_event({"stage": "video_info", "fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration})

            # Build sample indices (safe-guard)
            max_samples = min(30, max(1, frame_count))
            indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))

            captioner = get_captioner_for_model(model)
            if captioner is None:
                yield _sse_event({"stage": "error", "message": f"no local captioner available for model '{model}'"})
                return

            # Open second capture for seeking samples
            cap2 = cv2.VideoCapture(file_path)
            for fi in indices:
                try:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap2.read()
                    if not ret:
                        yield _sse_event({"stage": "sample_error", "frame_index": fi, "error": "read_failed"})
                        continue
                    # convert BGR->RGB PIL image
                    from PIL import Image
                    import numpy as np
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    out = captioner(img)
                    caption = _normalize_caption_output(captioner, out)

                    # Decide label: optionally via LLM
                    label = 'idle'
                    if use_llm:
                        text_llm = get_local_text_llm()
                        if text_llm is not None:
                            try:
                                cls_prompt = CLASSIFY_PROMPT_TEMPLATE.format(prompt=prompt, caption=caption)
                                cls_out = text_llm(cls_prompt, max_new_tokens=8)
                                cls_text = None
                                if isinstance(cls_out, list) and len(cls_out) > 0:
                                    f0 = cls_out[0]
                                    if isinstance(f0, dict):
                                        cls_text = f0.get('generated_text') or f0.get('text') or str(f0)
                                    else:
                                        cls_text = str(f0)
                                else:
                                    cls_text = str(cls_out)
                                if cls_text:
                                    ct = cls_text.lower()
                                    if 'work' in ct and 'idle' not in ct:
                                        label = 'work'
                                    elif 'idle' in ct and 'work' not in ct:
                                        label = 'idle'
                            except Exception as e:
                                logging.info('SSE LLM classification failed: %s', e)
                    if label == 'idle':
                        lw = caption.lower() if isinstance(caption, str) else ''
                        work_keywords = ['work', 'welding', 'screw', 'screwing', 'tool', 'using', 'assemble', 'hands', 'holding']
                        if any(w in lw for w in work_keywords):
                            label = 'work'

                    time_sec = fi / (fps if fps > 0 else 30.0)
                    yield _sse_event({"stage": "sample", "frame_index": fi, "time_sec": time_sec, "caption": caption, "label": label})
                except Exception as e:
                    yield _sse_event({"stage": "sample_error", "frame_index": fi, "error": str(e)})
            cap2.release()

            yield _sse_event({"stage": "finished", "message": "processing complete", "video_url": f"/backend/vlm_video/{filename}"})
        except Exception as e:
            yield _sse_event({"stage": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post("/backend/vlm_local")
async def vlm_local(
    model: str = Form(...),
    prompt: str = Form(''),
    video: UploadFile = File(None),
    use_llm: bool = Form(False),
):
    """Run a local lightweight VLM-like captioner on a representative frame.
    This uses a BLIP image->text pipeline (Salesforce/blip-image-captioning-large) if installed.
    Returns the caption(s) plus the same basic video analysis as `/backend/vlm`.
    """
    analysis = {"model": model, "prompt": prompt}

    if video is None:
        raise HTTPException(status_code=400, detail="No video provided")

    # Save uploaded video
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{video.filename}"
    save_path = os.path.join(VLM_UPLOAD_DIR, filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Basic video analysis (same as /vlm)
    try:
        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            analysis.update({"warning": "could not open saved video"})
            return {"message": "VLM local failed", "analysis": analysis}

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0

        # Choose a representative frame (middle)
        mid_frame = int(frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            avg_color_per_row = frame.mean(axis=0).mean(axis=0)
            avg_color = [float(avg_color_per_row[2]), float(avg_color_per_row[1]), float(avg_color_per_row[0])]
            analysis.update({
                "width": width,
                "height": height,
                "fps": float(fps),
                "frame_count": int(frame_count),
                "duration_sec": float(duration),
                "first_frame_avg_color_bgr": avg_color,
            })
        else:
            analysis.update({"warning": "could not read mid frame"})

        cap.release()
    except Exception as e:
        analysis.update({"error": str(e)})

    # Run local captioner on multiple sampled frames if available (respect requested model)
    captioner = get_captioner_for_model(model)
    if captioner is None:
        analysis.update({"local_captioner": "not available (install transformers/torch or cache model)"})
        analysis["video_url"] = f"/backend/vlm_video/{filename}"
        return {"message": "VLM local stub response", "analysis": analysis}

    try:
        import cv2 as _cv2
        from PIL import Image

        # Decide sampling strategy: up to `max_samples` evenly across the video
        max_samples = 30
        total_frames = int(frame_count)
        desired = min(max_samples, max(1, total_frames))
        step = max(1, total_frames // desired) if total_frames > 0 else 1

        samples = []
        work_frames = []
        idle_frames = []

        # simple keyword-based classifier for 'work' vs 'idle'
        work_keywords = [
            'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
            'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
            'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
        ]

        cap2 = cv2.VideoCapture(save_path)
        if not cap2.isOpened():
            analysis.update({"error": "could not open video for sampling"})
            analysis["video_url"] = f"/backend/vlm_video/{filename}"
            return {"message": "VLM local response", "analysis": analysis}

        for idx in range(0, total_frames, step):
            if len(samples) >= max_samples:
                break

            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_s, f = cap2.read()
            if not ret_s or f is None:
                continue

            # Convert to RGB PIL image and caption
            rgb = _cv2.cvtColor(f, _cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                out = captioner(pil)
            except Exception as ex:
                samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "caption": None, "error": str(ex)})
                idle_frames.append(idx)
                continue

            # normalize output to text
            text = _normalize_caption_output(captioner, out)

            text_lower = (text or '').lower()
            is_work = any(k in text_lower for k in work_keywords)

            samples.append({
                "frame_index": idx,
                "time_sec": float(idx / fps) if fps > 0 else 0.0,
                "caption": text,
                "label": "work" if is_work else "idle",
            })

            if is_work:
                work_frames.append(idx)
            else:
                idle_frames.append(idx)

        cap2.release()

        analysis["samples"] = samples
        analysis["idle_frames"] = idle_frames
        analysis["work_frames"] = work_frames

    except Exception as e:
        analysis.update({"caption_error": str(e)})

    analysis["video_url"] = f"/backend/vlm_video/{filename}"
    return {"message": "VLM local response", "analysis": analysis}


@app.get("/backend/vlm_video/{filename}")
async def get_vlm_video(filename: str):
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/backend/vlm_local_models")
async def vlm_local_models():
    """Return a list of locally-available VLM/image-captioning models (if any).
    Probes model entries listed in `local_vlm_models.json` using
    `transformers.pipeline(..., local_files_only=True)` so no downloads occur.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
    models_found = []
    try:
        import json
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            entries = cfg.get('models', [])
    except Exception:
        entries = []

    # Try to probe each entry; do not trigger downloads (local_files_only=True)
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        # transformers/torch not installed; return entries but mark unavailable
        return {"available": False, "models": [{"id": e.get('id'), "name": e.get('name'), "available": False, "error": 'transformers or torch not installed'} for e in entries]}

    logging.info(f"Probing {len(entries)} local VLM models...")
    for e in entries:
        # print("entry:", e)
        logging.info(f"Probing Entry {e}")
        mid = e.get('id')
        task = e.get('task', 'image-to-text')
        info = {"id": mid, "name": e.get('name'), "available": False}
        try:
            # local_files_only prevents downloads and will raise if missing
            _p = pipeline(task, model=mid, device=device, local_files_only=True)
            info['available'] = True
        except Exception as ex:
            info['available'] = False
            info['error'] = str(ex)
        models_found.append(info)

    return {"available": any(m.get('available') for m in models_found), "models": models_found}


@app.post('/backend/llm_length_check')
async def llm_length_check(text: str = Form(...), max_context: int = Form(2048)):
    """Check whether `text` is too long for a model with `max_context` tokens.
    Attempts to use a local text LLM to give a suggested summary or strategy; falls
    back to a heuristic token estimate if no local LLM is available.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail='No text provided')

    # rough token estimate: approx 1 token ~= 4 characters (very approximate)
    approx_tokens = max(1, int(len(text) / 4))
    too_long = approx_tokens > int(max_context)

    suggestions = []
    if too_long:
        suggestions.append({"type": "summarize", "desc": f"Summarize the text to ~{int(max_context * 0.25)} tokens (short summary)."})
        suggestions.append({"type": "split", "desc": f"Split the text into {max(2, approx_tokens // int(max_context))} segments each <= {max_context} tokens."})
        suggestions.append({"type": "extract_actions", "desc": "Extract action items or steps only (reduce verbose context)."})
    else:
        suggestions.append({"type": "ok", "desc": "Text fits within the context window; no change required."})

    # Try to use a local text LLM to produce a short example summary/proposal
    example_summary = None
    text_llm = get_local_text_llm()
    if text_llm is not None:
        try:
            prompt = (
                f"You are an assistant that evaluates whether a text is too long for a model with "
                f"context {max_context} tokens. Respond with a 1-2 sentence summary and a single suggestion.\n\nText:\n" + text
            )
            # some pipelines return list/dict; call safely
            out = text_llm(prompt, max_new_tokens=128)
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    example_summary = first.get('generated_text') or first.get('text') or str(first)
                else:
                    example_summary = str(first)
            else:
                example_summary = str(out)
        except Exception as e:
            logging.info('Local text LLM failed: %s', e)

    return {
        "too_long": bool(too_long),
        "approx_tokens": int(approx_tokens),
        "max_context": int(max_context),
        "suggestions": suggestions,
        "example_summary": example_summary,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

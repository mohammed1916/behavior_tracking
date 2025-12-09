from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import uuid
from typing import Optional, Callable
from fastapi import Query
import json
import logging
import subprocess
from transformers import pipeline
import torch
import sqlite3
import time
from PIL import Image
import numpy as np
import re
import base64
import io
from datetime import datetime

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
# This prompt instructs the model to reply with exactly one lowercase word: either
# "work" or "idle" and nothing else. Keep responses machine-parsable.
CLASSIFY_PROMPT_TEMPLATE = (
    "You are a strict classifier. Respond with exactly one lowercase word: either work or idle.\n"
    "Do NOT add any punctuation, explanation, quotes, or extra text — only the single word.\n\n"
    "Context: {prompt}\nCaption: {caption}\n"
)

# Prompt for duration estimation: ask the LLM to return a single integer (seconds)
DURATION_PROMPT_TEMPLATE = (
    "You are an assistant that estimates how long a described manual task typically takes.\n"
    "Given the task description below, respond with a single integer representing the estimated time in seconds.\n"
    "Do NOT add any explanation or text — only the integer number of seconds.\n\n"
    "Task: {task}\n"
)

# Prompt for task completion evaluation
TASK_COMPLETION_PROMPT_TEMPLATE = (
    "You are a task completion evaluator. Based on the captions from video analysis, determine if the specified task has been completed.\n"
    "Respond with exactly one word: either 'yes' or 'no'.\n"
    "Do NOT add any punctuation, explanation, quotes, or extra text — only the single word.\n\n"
    "Task: {task}\nCaptions: {captions}\n"
)

# Cache captioners by model id so repeated requests don't reload them
_captioner_cache = {}

def get_local_captioner():
    """Return a default local captioner (BLIP) if available."""
    global _local_captioner
    if _local_captioner is not None:
        return _local_captioner

    device = 0 if torch.cuda.is_available() else -1
    try:
        # _local_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device, local_files_only=True)
        _local_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
        logging.info('Loaded default local captioner Salesforce/blip-image-captioning-large')
    except Exception:
        _local_captioner = None
    return _local_captioner


def get_captioner_for_model(model_id: str):
    """Return a captioner pipeline for `model_id`.
    Tries local cache first (`local_files_only=True`), then falls back to downloading the model.
    Special handling for OpenFlamingo models.
    """
    global _captioner_cache
    if not model_id:
        return get_local_captioner()

    if model_id in _captioner_cache:
        # For OpenFlamingo models, retry if None cached (might have been fixed)
        if ('openflamingo' in model_id.lower() or 'flamingo' in model_id.lower()) and _captioner_cache[model_id] is None:
            logging.info(f'Retrying previously failed OpenFlamingo model: {model_id}')
            del _captioner_cache[model_id]  # Clear the None cache
        else:
            return _captioner_cache[model_id]

    # OpenFlamingo support removed: fall through to normal pipeline loading
    # (Previously there was special-case OpenFlamingo handling here.)

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
    cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        for e in cfg.get('models', []):
            if e.get('id') == model_id:
                task = e.get('task', task)
                break
 
    p = pipeline(task, model=model_id, device=device)
    _captioner_cache[model_id] = p
    logging.info('Loaded captioner for model %s (task=%s) from local cache on device=%s', model_id, task, device)
    return p




   


def _get_flamingo_captioner(model_id: str):
    """OpenFlamingo support removed.

    This server no longer includes special-case OpenFlamingo loading logic.
    Keep a small stub so any legacy callers get a clear message.
    """
    logging.info('OpenFlamingo support removed from server; use BLIP or other HF models')
    _captioner_cache[model_id] = None
    return None


# Lazy loader for a local text LLM 
_local_text_llm = None
def get_local_text_llm():
    global _local_text_llm
    if _local_text_llm is not None:
        return _local_text_llm

    if shutil.which('ollama'):
        # ollama_model = os.environ.get('OLLAMA_MODEL', 'strat') 
        ollama_model = 'qwen3:0.6b'
        class OllamaWrapper:
            def __init__(self, model_id):
                self.model_id = model_id
            def __call__(self, prompt, max_new_tokens=128):
                try:
                    # Capture raw bytes and decode explicitly with UTF-8 and
                    # replacement to avoid platform-default cp1252 decoding errors
                    p = subprocess.run(
                        ['ollama', 'run', self.model_id, prompt],
                        capture_output=True,
                        timeout=120,
                    )
                    out_bytes = p.stdout or b''
                    try:
                        out = out_bytes.decode('utf-8', errors='replace').strip()
                        logging.info('Ollama decoded output: %s', out)
                    except Exception:
                        logging.info('Ollama output decoding failed, using raw bytes string')
                        out = str(out_bytes)
                        logging.info('Ollama raw output string: %s', out)   
                    return [{'generated_text': out}]
                except Exception as e:
                    logging.info('Ollama run failed: %s', e)
                    return [{'generated_text': ''}]
        _local_text_llm = OllamaWrapper(ollama_model)
        logging.info('Using Ollama CLI model %s for text LLM', ollama_model)
        return _local_text_llm


    # try:
    #     from transformers import pipeline
    #     import torch
    # except Exception:
    #     _local_text_llm = None
    #     return None
    # device = 0 if torch.cuda.is_available() else -1
    # # Try a small local-capable HF model first (must be cached locally to avoid downloads)
    # candidates = [
    #     'Qwen/Qwen-2.5',
    #     'qwen/Qwen-2.5',
    #     'gpt2',
    # ]
    # for cid in candidates:
    #     try:
    #         _local_text_llm = pipeline('text-generation', model=cid, device=device, local_files_only=True)
    #         return _local_text_llm
    #     except Exception:
    #         _local_text_llm = None
    # return None


def _normalize_caption_output(captioner, out):
    """Normalize various pipeline outputs into a text string.
    Some models/pipelines may return dicts containing token ids (e.g. 'input_ids').
    Try to decode those using the pipeline's tokenizer when possible.
    Handles OpenFlamingo wrapper outputs as well.
    """
    try:
        first = None
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
        else:
            first = out

        try:
            # common keys
            for k in ('generated_text', 'caption', 'text'):
                if hasattr(first, 'get') and first.get(k):
                    return first.get(k)

            # If token ids are present, try to decode via tokenizer
            if hasattr(first, '__contains__') and 'input_ids' in first:
                ids = first.get('input_ids') if hasattr(first, 'get') else None
                if ids is not None:
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

                # Handle OpenFlamingo processor decoding
                if ids_list and hasattr(captioner, 'tokenizer') and hasattr(captioner.tokenizer, 'decode'):
                    try:
                        return captioner.tokenizer.decode(ids_list, skip_special_tokens=True)
                    except Exception:
                        pass

            return str(first)
        except Exception:
            return str(first)
    except Exception:
        logging.exception("Error normalizing caption output")
        return "Error normalizing caption output"



# def process_video_samples(file_path: str, captioner=None, max_samples: int = 30, stream_callback: Optional[Callable] = None, prompt: str = '', use_llm: bool = False):
#     """Process a video file: compute basic video_info and sample frames to caption.

#     Args:
#         file_path (str): Path to the video file.
#         captioner: Optional captioning model to use for frame captions.
#         max_samples (int): Maximum number of frames to sample and caption.
#         stream_callback (Optional[Callable]): If provided, called with event dicts for each stage/sample.
#         prompt (str): Optional prompt to guide the captioning or LLM, if applicable.
#         use_llm (bool): If True, use a language model (LLM) for captioning or analysis.

#     Returns:
#         dict: A dictionary with video_info, samples, idle_frames, work_frames.
#     """
#     result = {}
#     try:
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             msg = {"stage": "error", "message": "could not open video"}
#             if stream_callback:
#                 stream_callback(msg)
#             else:
#                 result.update({"error": "could not open video"})
#             return result

#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#         fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
#         duration = frame_count / (fps if fps > 0 else 30.0)

#         video_info = {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}
#         if stream_callback:
#             stream_callback({"stage": "video_info", **video_info})

#         total_frames = max(1, frame_count)
#         desired = min(max_samples, max(1, total_frames))
#         step = max(1, total_frames // desired) if total_frames > 0 else 1

#         samples = []
#         work_frames = []
#         idle_frames = []

#         work_keywords = [
#             'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
#             'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
#             'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
#         ]

#         text_llm = None
#         if use_llm:
#             try:
#                 text_llm = get_local_text_llm()
#                 if text_llm is None:
#                     logging.info('LLM classification requested but no local text LLM available')
#             except Exception:
#                 text_llm = None

#         if captioner is None:
#             captioner = get_local_captioner()

#         if captioner is None:
#             msg = {"stage": "error", "message": "no local captioner available"}
#             if stream_callback:
#                 stream_callback(msg)
#             else:
#                 result.update({"local_captioner": "not available"})
#             cap.release()
#             return result

#         cap2 = cv2.VideoCapture(file_path)
#         if not cap2.isOpened():
#             msg = {"stage": "error", "message": "could not open video for sampling"}
#             if stream_callback:
#                 stream_callback(msg)
#             else:
#                 result.update({"error": "could not open video for sampling"})
#             cap.release()
#             return result

#         for idx in range(0, int(total_frames), step):
#             if len(samples) >= max_samples:
#                 break
#             try:
#                 cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
#                 ret, f = cap2.read()
#                 if not ret or f is None:
#                     if stream_callback:
#                         stream_callback({"stage": "sample_error", "frame_index": idx, "error": "read_failed"})
#                     else:
#                         samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": "read_failed"})
#                     continue

#                 # caption
#                 rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#                 pil = Image.fromarray(rgb)
#                 try:
#                     out = captioner(pil)
#                 except Exception as ex:
#                     if stream_callback:
#                         stream_callback({"stage": "sample_error", "frame_index": idx, "error": str(ex)})
#                     else:
#                         samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": str(ex)})
#                         idle_frames.append(idx)
#                     continue

#                 text = _normalize_caption_output(captioner, out)

#                 # Decide label: prefer LLM-based classification if requested and available,
#                 # otherwise fall back to simple keyword matching. Also capture LLM output
#                 # into the sample so it appears in the returned `analysis`.
#                 is_work = False
#                 llm_output = None
#                 if use_llm and text_llm is not None:
#                     try:
#                         cls_prompt = CLASSIFY_PROMPT_TEMPLATE.format(prompt=prompt, caption=text)
#                         llm_raw = text_llm(cls_prompt, max_new_tokens=8)
#                         out_text = None
#                         if isinstance(llm_raw, list) and len(llm_raw) > 0:
#                             first = llm_raw[0]
#                             if isinstance(first, dict):
#                                 out_text = first.get('generated_text') or first.get('text') or str(first)
#                             else:
#                                 out_text = str(first)
#                         else:
#                             out_text = str(llm_raw)

#                         llm_output = out_text

#                         if out_text:
#                             out_text_l = out_text.lower()
#                             if 'work' in out_text_l and 'idle' not in out_text_l:
#                                 is_work = True
#                             elif 'idle' in out_text_l and 'work' not in out_text_l:
#                                 is_work = False
#                             else:
#                                 # ambiguous -> fallback to keyword
#                                 is_work = any(k in (text or '').lower() for k in work_keywords)
#                     except Exception as e:
#                         logging.info('LLM classifier failed: %s', e)
#                         is_work = any(k in (text or '').lower() for k in work_keywords)
#                         llm_output = None
#                 else:
#                     text_lower = (text or '').lower()
#                     is_work = any(k in text_lower for k in work_keywords)

#                 sample = {"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "caption": text, "label": "work" if is_work else "idle", "llm_output": llm_output}
#                 if stream_callback:
#                     stream_callback({"stage": "sample", **sample})
#                 else:
#                     samples.append(sample)
#                     if is_work:
#                         work_frames.append(idx)
#                     else:
#                         idle_frames.append(idx)
#             except Exception as e:
#                 if stream_callback:
#                     stream_callback({"stage": "sample_error", "frame_index": idx, "error": str(e)})
#                 else:
#                     samples.append({"frame_index": idx, "time_sec": float(idx / fps) if fps > 0 else 0.0, "error": str(e)})

#         cap2.release()
#         cap.release()

#         if not stream_callback:
#             result.update({"video_info": video_info, "samples": samples, "idle_frames": idle_frames, "work_frames": work_frames})
#         return result
#     except Exception as e:
#         if stream_callback:
#             stream_callback({"stage": "error", "message": str(e)})
#             return {}
#         return {"error": str(e)}

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

# SQLite DB setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'analyses.db')
_db_initialized = False

def init_db():
    global _db_initialized
    if _db_initialized:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        filename TEXT,
        model TEXT,
        prompt TEXT,
        video_url TEXT,
        fps REAL,
        frame_count INTEGER,
        width INTEGER,
        height INTEGER,
        duration REAL,
        subtask_id TEXT,
        created_at TEXT
    )
    ''')
    # try:
    #     cur.execute('ALTER TABLE analyses ADD COLUMN subtask_id TEXT;')
    # except sqlite3.OperationalError:
    #     pass  # Column already exists
    cur.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS subtasks (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        subtask_info TEXT,
        duration_sec INTEGER NOT NULL,
        completed_in_time INTEGER DEFAULT 0,
        completed_with_delay INTEGER DEFAULT 0,
        created_at TEXT,
        FOREIGN KEY (task_id) REFERENCES tasks(id)
    )
    ''')
    try:
        cur.execute('ALTER TABLE subtasks ADD COLUMN completed_in_time INTEGER DEFAULT 0;')
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cur.execute('ALTER TABLE subtasks ADD COLUMN completed_with_delay INTEGER DEFAULT 0;')
    except sqlite3.OperationalError:
        pass  # Column already exists
    cur.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT,
        frame_index INTEGER,
        time_sec REAL,
        caption TEXT,
        label TEXT,
        llm_output TEXT,
        FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    )
    ''')
    conn.commit()
    conn.close()
    _db_initialized = True

def save_analysis_to_db(analysis_id, filename, model, prompt, video_url, video_info, samples, subtask_id=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created = datetime.utcnow().isoformat() + 'Z'
    cur.execute('''INSERT OR REPLACE INTO analyses (id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        analysis_id, filename, model, prompt, video_url,
        video_info.get('fps') if video_info else None,
        video_info.get('frame_count') if video_info else None,
        video_info.get('width') if video_info else None,
        video_info.get('height') if video_info else None,
        video_info.get('duration') if video_info else None,
        subtask_id,
        created
    ))

    # insert samples
    if samples:
        for s in samples:
            cur.execute('''INSERT INTO samples (analysis_id, frame_index, time_sec, caption, label, llm_output) VALUES (?, ?, ?, ?, ?, ?)''', (
                analysis_id,
                s.get('frame_index'),
                s.get('time_sec'),
                s.get('caption'),
                s.get('label'),
                s.get('llm_output')
            ))

    conn.commit()
    conn.close()

def list_analyses_from_db(limit=100):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at FROM analyses ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'id': r[0], 'filename': r[1], 'model': r[2], 'prompt': r[3], 'video_url': r[4],
            'fps': r[5], 'frame_count': r[6], 'width': r[7], 'height': r[8], 'duration': r[9],
            'subtask_id': r[10], 'created_at': r[11]
        })
    return out

def get_analysis_from_db(aid):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at FROM analyses WHERE id=?', (aid,))
    r = cur.fetchone()
    if not r:
        conn.close()
        return None
    analysis = {
        'id': r[0], 'filename': r[1], 'model': r[2], 'prompt': r[3], 'video_url': r[4],
        'fps': r[5], 'frame_count': r[6], 'width': r[7], 'height': r[8], 'duration': r[9],
        'subtask_id': r[10], 'created_at': r[11]
    }
    cur.execute('SELECT frame_index, time_sec, caption, label, llm_output FROM samples WHERE analysis_id=? ORDER BY frame_index ASC', (aid,))
    rows = cur.fetchall()
    samples = []
    idle_frames = []
    work_frames = []
    for s in rows:
        sample = {'frame_index': s[0], 'time_sec': s[1], 'caption': s[2], 'label': s[3], 'llm_output': s[4]}
        samples.append(sample)
        if s[3] == 'idle':
            idle_frames.append(s[0])
        elif s[3] == 'work':
            work_frames.append(s[0])
    analysis['samples'] = samples
    analysis['idle_frames'] = idle_frames
    analysis['work_frames'] = work_frames
    conn.close()
    return analysis

def delete_analysis_from_db(aid):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('DELETE FROM samples WHERE analysis_id=?', (aid,))
    cur.execute('DELETE FROM analyses WHERE id=?', (aid,))
    conn.commit()
    conn.close()

def save_task_to_db(task_id, name):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created = datetime.utcnow().isoformat() + 'Z'
    cur.execute('''INSERT OR REPLACE INTO tasks (id, name, created_at)
                   VALUES (?, ?, ?)''', (
        task_id, name, created
    ))
    conn.commit()
    conn.close()

def list_tasks_from_db():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, created_at FROM tasks ORDER BY created_at DESC')
    rows = cur.fetchall()
    conn.close()
    return [{
        'id': r[0], 'name': r[1], 'created_at': r[2]
    } for r in rows]

def get_task_from_db(task_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, created_at FROM tasks WHERE id=?', (task_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        'id': r[0], 'name': r[1], 'created_at': r[2]
    }

def save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time=0, completed_with_delay=0):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created = datetime.utcnow().isoformat() + 'Z'
    cur.execute('''INSERT OR REPLACE INTO subtasks (id, task_id, subtask_info, duration_sec, completed_in_time, completed_with_delay, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''', (
        subtask_id, task_id, subtask_info, duration_sec, completed_in_time, completed_with_delay, created
    ))
    conn.commit()
    conn.close()

def list_subtasks_from_db():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT s.id, t.name, s.subtask_info, s.duration_sec, s.completed_in_time, s.completed_with_delay, s.created_at
        FROM subtasks s
        JOIN tasks t ON s.task_id = t.id
        ORDER BY s.created_at DESC
    ''')
    rows = cur.fetchall()
    conn.close()
    return [{
        'id': r[0], 'task_name': r[1], 'subtask_info': r[2], 'duration_sec': r[3], 'completed_in_time': r[4], 'completed_with_delay': r[5], 'created_at': r[6]
    } for r in rows]

def get_subtask_from_db(subtask_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        SELECT s.id, t.name, s.subtask_info, s.duration_sec, s.completed_in_time, s.completed_with_delay, s.created_at
        FROM subtasks s
        JOIN tasks t ON s.task_id = t.id
        WHERE s.id = ?
    ''', (subtask_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        'id': r[0], 'task_name': r[1], 'subtask_info': r[2], 'duration_sec': r[3], 'completed_in_time': r[4], 'completed_with_delay': r[5], 'created_at': r[6]
    }

def compute_ranges(frames, samples, fps):
    if not frames or frames.length == 0:
        return []
    sample_map = {}
    for s in samples or []:
        if s and 'frame_index' in s and 'time_sec' in s:
            sample_map[s['frame_index']] = s['time_sec']
    times = []
    for f in frames:
        time_val = sample_map.get(f, (f / fps) if fps else 0)
        times.append({'frame': f, 'time': time_val})
    times.sort(key=lambda x: x['time'])
    dt_est = 1.0
    if samples and len(samples) >= 2:
        s0 = samples[0].get('time_sec', 0)
        s1 = samples[1].get('time_sec', 0)
        d = abs(s1 - s0)
        if d > 0:
            dt_est = d
    max_gap = max(1.0, dt_est * 1.5)
    ranges = []
    if times:
        cur = {'startFrame': times[0]['frame'], 'endFrame': times[0]['frame'], 'startTime': times[0]['time'], 'endTime': times[0]['time']}
        for i in range(1, len(times)):
            t = times[i]
            if (t['time'] - cur['endTime']) <= max_gap:
                cur['endFrame'] = t['frame']
                cur['endTime'] = t['time']
            else:
                ranges.append(cur.copy())
                cur = {'startFrame': t['frame'], 'endFrame': t['frame'], 'startTime': t['time'], 'endTime': t['time']}
        ranges.append(cur)
    # Add captions to ranges
    for r in ranges:
        captions = [s['caption'] for s in (samples or []) if 'time_sec' in s and r['startTime'] - 0.0001 <= s['time_sec'] <= r['endTime'] + 0.0001 and s.get('caption')]
        r['captions'] = captions
    return ranges

@app.get("/backend/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename, headers={"Accept-Ranges": "bytes"})
    raise HTTPException(status_code=404, detail="File not found")

# Global flag to control webcam streaming
_webcam_active = False

@app.get("/backend/stream_pose")
async def stream_pose(model: str = Query(''), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), compare_timings: bool = Query(False)):
    """Stream webcam frames as SSE events with BLIP captioning and optional LLM classification every 2 seconds.
    Emits structured payloads similar to /vlm_local_stream, and saves analysis to DB at the end.
    """
    global _webcam_active
    _webcam_active = True
    def gen():
        try:
            yield _sse_event({"stage": "started", "message": "live processing started"})
            
            cap = cv2.VideoCapture(0)
            captioner = get_captioner_for_model(model) if model else get_local_captioner()
            if captioner is None:
                yield _sse_event({"stage": "error", "message": "no captioner available"})
                return
            
            current_caption = "Initializing..."
            last_inference_time = time.time()
            frame_counter = 0
            collected_samples = []
            collected_idle = []
            collected_work = []
            cumulative_work_frames = 0
            start_time = time.time()
            
            while _webcam_active:
                ret, frame = cap.read()
                if not ret:
                    yield _sse_event({"stage": "error", "message": "failed to read frame"})
                    break
                frame = cv2.flip(frame, 1)
                frame_counter += 1
                elapsed_time = time.time() - start_time
                
                if time.time() - last_inference_time >= 2.0:
                    try:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        out = captioner(img)
                        caption = _normalize_caption_output(captioner, out)
                        
                        # Classification logic (same as vlm_local_stream)
                        label = 'idle'
                        cls_text = None
                        if use_llm:
                            text_llm = get_local_text_llm()
                            if text_llm is not None:
                                try:
                                    cls_prompt = CLASSIFY_PROMPT_TEMPLATE.format(prompt=prompt, caption=caption)
                                    cls_out = text_llm(cls_prompt, max_new_tokens=8)
                                    if isinstance(cls_out, list) and len(cls_out) > 0:
                                        f0 = cls_out[0]
                                        if isinstance(f0, dict):
                                            cls_text = f0.get('generated_text') or f0.get('text') or str(f0)
                                        else:
                                            cls_text = str(f0)
                                    else:
                                        cls_text = str(cls_out)
                                    if cls_text:
                                        cleaned = re.sub(r'[^\w\s]', '', cls_text).strip()
                                        words = cleaned.split()
                                        if words:
                                            last_word = words[-1].lower()
                                            if last_word == 'work':
                                                label = 'work'
                                            elif last_word == 'idle':
                                                label = 'idle'
                                except Exception as e:
                                    logging.info('SSE LLM classification failed: %s', e)
                        if label == 'idle':
                            lw = caption.lower() if isinstance(caption, str) else ''
                            work_keywords = [
                                'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
                                'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
                                'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
                            ]
                            if any(w in lw for w in work_keywords):
                                label = 'work'
                        
                        # Subtask overrun check (same as vlm_local_stream)
                        subtask_overrun = None
                        if subtask_id and label == 'work':
                            cumulative_work_frames += 1
                            try:
                                assign = get_subtask_from_db(subtask_id)
                                if assign is not None and assign.get('duration_sec') is not None:
                                    expected = assign.get('duration_sec')
                                    actual_work_time = cumulative_work_frames / 30.0  # Assume 30 FPS for live stream
                                    if actual_work_time > expected:
                                        subtask_overrun = True
                                    else:
                                        subtask_overrun = False
                            except Exception:
                                subtask_overrun = None
                        
                        payload = {"stage": "sample", "frame_index": frame_counter, "time_sec": elapsed_time, "caption": caption, "label": label, "llm_output": cls_text}
                        if subtask_overrun is not None:
                            payload['subtask_overrun'] = subtask_overrun
                        yield _sse_event(payload)
                        
                        # Collect for DB saving
                        collected_samples.append({
                            'frame_index': frame_counter, 'time_sec': elapsed_time, 'caption': caption, 'label': label, 'llm_output': cls_text
                        })
                        if label == 'work':
                            collected_work.append(frame_counter)
                        else:
                            collected_idle.append(frame_counter)
                        
                        last_inference_time = time.time()
                    except Exception as e:
                        yield _sse_event({"stage": "sample_error", "frame_index": frame_counter, "error": str(e)})
            
            cap.release()
            
            # Save to DB at the end (similar to vlm_local_stream)
            try:
                vid_info = {'fps': 30.0, 'frame_count': frame_counter, 'width': 640, 'height': 480, 'duration': elapsed_time}  # Approximate for live stream
                aid = str(uuid.uuid4())
                save_analysis_to_db(aid, f"live_webcam_{aid}.mp4", model or 'default', prompt, None, vid_info, collected_samples, subtask_id=subtask_id)
                
                # Update subtask counts if compare_timings enabled
                if compare_timings and subtask_id:
                    try:
                        subtask = get_subtask_from_db(subtask_id)
                        if subtask and collected_work:
                            expected_duration = subtask['duration_sec']
                            min_frame = min(collected_work)
                            max_frame = max(collected_work)
                            actual_work_time = (max_frame - min_frame) / 30.0  # Assume 30 FPS
                            if actual_work_time <= expected_duration:
                                update_subtask_counts(subtask_id, 1, 0)
                            else:
                                update_subtask_counts(subtask_id, 0, 1)
                    except Exception as e:
                        logging.exception(f'Failed to update subtask counts for {subtask_id}')
                
                yield _sse_event({"stage": "finished", "message": "live processing complete", "stored_analysis_id": aid})
            except Exception as e:
                yield _sse_event({"stage": "finished", "message": "live processing complete"})
        except Exception as e:
            yield _sse_event({"stage": "error", "message": str(e)})
        finally:
            _webcam_active = False
    
    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post("/backend/stop_webcam")
async def stop_webcam():
    """Stop the webcam stream."""
    global _webcam_active
    _webcam_active = False
    return {"message": "Webcam stopped"}


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
async def vlm_local_stream(filename: str = Query(...), model: str = Query(...), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), compare_timings: bool = Query(False)):
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
            # Track cumulative work frames when subtask monitoring is requested
            cumulative_work_frames = 0
            # Collect final samples to allow storing into DB at the end
            collected_samples = []
            collected_idle = []
            collected_work = []
            for fi in indices:
                try:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ret, frame = cap2.read()
                    if not ret:
                        yield _sse_event({"stage": "sample_error", "frame_index": fi, "error": "read_failed"})
                        continue
                    # convert BGR->RGB PIL image

                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    out = captioner(img)
                    caption = _normalize_caption_output(captioner, out)

                    # Decide label: optionally via LLM
                    label = 'idle'
                    cls_text = None
                    if use_llm:
                        text_llm = get_local_text_llm()
                        if text_llm is not None:
                            try:
                                cls_prompt = CLASSIFY_PROMPT_TEMPLATE.format(prompt=prompt, caption=caption)
                                cls_out = text_llm(cls_prompt, max_new_tokens=8)
                                if isinstance(cls_out, list) and len(cls_out) > 0:
                                    f0 = cls_out[0]
                                    if isinstance(f0, dict):
                                        cls_text = f0.get('generated_text') or f0.get('text') or str(f0)
                                    else:
                                        cls_text = str(f0)
                                else:
                                    cls_text = str(cls_out)
                                if cls_text:
                                    cleaned = re.sub(r'[^\w\s]', '', cls_text).strip()
                                    words = cleaned.split()
                                    if words:
                                        last_word = words[-1].lower()
                                        if last_word == 'work':
                                            label = 'work'
                                        elif last_word == 'idle':
                                            label = 'idle'
                            except Exception as e:
                                logging.info('SSE LLM classification failed: %s', e)
                    if label == 'idle':
                        lw = caption.lower() if isinstance(caption, str) else ''
                        # work_keywords = ['work', 'welding', 'screw', 'screwing', 'tool', 'using', 'assemble', 'hands', 'holding']
                        work_keywords = [
                            'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
                            'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
                            'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
                        ]
                        if any(w in lw for w in work_keywords):
                            label = 'work'

                    time_sec = fi / (fps if fps > 0 else 30.0)

                    # If monitoring a subtask, compute cumulative work time and emit a flag event when exceeded
                    subtask_overrun = None
                    if subtask_id and label == 'work':
                        cumulative_work_frames += 1
                        try:
                            assign = get_subtask_from_db(subtask_id)
                            if assign is not None and assign.get('expected_duration_sec') is not None:
                                expected = assign.get('expected_duration_sec')
                                actual_work_time = cumulative_work_frames / (fps if fps > 0 else 30.0)
                                if actual_work_time > expected:
                                    subtask_overrun = True
                                else:
                                    subtask_overrun = False
                        except Exception:
                            subtask_overrun = None

                    payload = {"stage": "sample", "frame_index": fi, "time_sec": time_sec, "caption": caption, "label": label, "llm_output": cls_text}
                    if subtask_overrun is not None:
                        payload['subtask_overrun'] = subtask_overrun

                    # collect
                    collected_samples.append({
                        'frame_index': fi, 'time_sec': time_sec, 'caption': caption, 'label': label, 'llm_output': cls_text
                    })
                    if label == 'work':
                        collected_work.append(fi)
                    else:
                        collected_idle.append(fi)

                    yield _sse_event(payload)
                except Exception as e:
                    yield _sse_event({"stage": "sample_error", "frame_index": fi, "error": str(e)})
            cap2.release()

            # Persist the final collected analysis to DB
            try:
                vid_info = {'fps': fps, 'frame_count': frame_count, 'width': width, 'height': height, 'duration': duration}
                aid = str(uuid.uuid4())
                save_analysis_to_db(aid, filename, model, prompt, f"/backend/vlm_video/{filename}", vid_info, collected_samples, subtask_id=subtask_id)
                # Update subtask counts if compare_timings is enabled and subtask_id provided
                if compare_timings and subtask_id:
                    try:
                        subtask = get_subtask_from_db(subtask_id)
                        if subtask:
                            expected_duration = subtask['duration_sec']
                            # Calculate actual work time from collected_work (time span from first to last work frame)
                            if collected_work:
                                min_frame = min(collected_work)
                                max_frame = max(collected_work)
                                actual_work_time = (max_frame - min_frame) / fps if fps > 0 else 0
                                logging.info(f'Subtask {subtask_id}: expected={expected_duration}s, actual={actual_work_time}s, work_frames={len(collected_work)}')
                                if actual_work_time <= expected_duration:
                                    completed_in_time_increment = 1
                                    completed_with_delay_increment = 0
                                    logging.info(f'Incrementing completed_in_time for subtask {subtask_id}')
                                else:
                                    completed_in_time_increment = 0
                                    completed_with_delay_increment = 1
                                    logging.info(f'Incrementing completed_with_delay for subtask {subtask_id}')
                                # Update counts in database
                                update_subtask_counts(subtask_id, completed_in_time_increment, completed_with_delay_increment)
                                logging.info(f'Updated subtask {subtask_id} counts: +{completed_in_time_increment} in_time, +{completed_with_delay_increment} with_delay')
                            else:
                                logging.info(f'No work frames found for subtask {subtask_id}')
                    except Exception as e:
                        logging.exception(f'Failed to update subtask counts for {subtask_id}')
                yield _sse_event({"stage": "finished", "message": "processing complete", "video_url": f"/backend/vlm_video/{filename}", "stored_analysis_id": aid})
            except Exception:
                yield _sse_event({"stage": "finished", "message": "processing complete", "video_url": f"/backend/vlm_video/{filename}"})
        except Exception as e:
            yield _sse_event({"stage": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.get("/backend/vlm_video/{filename}")
async def get_vlm_video(filename: str):
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/backend/vlm_local_models")
async def vlm_local_models():
    """Return the list of candidate local VLM/image-captioning models without
    automatically instantiating heavy pipelines. This avoids loading/downloading
    models during probing; use `/backend/load_vlm_model` to load a model on demand.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), 'local_vlm_models.json')
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            entries = cfg.get('models', [])
    except Exception:
        entries = []

    models_out = []
    for e in entries:
        models_out.append({
            "id": e.get('id'),
            "name": e.get('name'),
            "task": e.get('task', 'image-to-text'),
            # We don't probe availability here to avoid loading models.
            "available": False,
            "probed": False,
        })

    return {"available": False, "models": models_out, "probed": False}


@app.post("/backend/caption")
async def caption_image(request: dict):
    """Caption an image using the specified VLM model.
    
    Expects JSON:
    {
        "image": "base64_encoded_image_data",
        "model": "model_id_string"
    }
    
    Returns:
    {
        "caption": "generated caption text",
        "model": "model_id_used"
    }
    """
    
    try:
        image_data = request.get("image")
        model_id = request.get("model")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="No model specified")
        
        # Decode base64 image
        try:
            # Handle data URL format (data:image/png;base64,...)
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed and ensure minimum size
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Ensure minimum size for BLIP (resize very small images)
            if image.size[0] < 224 or image.size[1] < 224:
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Get the appropriate captioner
        captioner = get_captioner_for_model(model_id)
        
        if captioner is None:
            raise HTTPException(status_code=500, detail=f"Model {model_id} is not available")
        
        # Generate caption
        try:
            result = captioner(image)
            
            # Extract caption text from result
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    caption_text = result[0]['generated_text']
                else:
                    caption_text = str(result[0])
            else:
                caption_text = str(result)
            
            return {
                "caption": caption_text,
                "model": model_id,
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"Error during caption generation with {model_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in caption endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post('/backend/load_vlm_model')
async def load_vlm_model(model: str = Form(...)):
    """Load and cache a VLM/captioner model on demand.
    Returns JSON indicating whether the model was successfully loaded.
    """
    if not model:
        raise HTTPException(status_code=400, detail='No model specified')
    try:
        p = get_captioner_for_model(model)
        if p is None:
            return {"loaded": False, "model": model, "error": "failed to load model (not available)"}
        return {"loaded": True, "model": model}
    except Exception as e:
        logging.exception('Error loading model %s', model)
        return {"loaded": False, "model": model, "error": str(e)}

@app.get('/backend/subtask/{subtask_id}')
async def get_subtask(subtask_id: str):
    s = get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    return s


@app.get('/backend/subtasks')
async def list_subtasks():
    return list_subtasks_from_db()


@app.post('/backend/subtasks')
async def create_subtask(
    task_id: str = Form(...),
    subtask_info: str = Form(''),
    duration_sec: int = Form(...),
):
    subtask_id = str(uuid.uuid4())
    save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec)
    return {"subtask_id": subtask_id}


@app.put('/backend/subtasks/{subtask_id}')
async def update_subtask(subtask_id: str, subtask_info: str = Form(...), duration_sec: int = Form(...), completed_in_time: int = Form(None), completed_with_delay: int = Form(None)):
    s = get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    # Get task_id from existing
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT task_id FROM subtasks WHERE id=?', (subtask_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        raise HTTPException(status_code=404, detail='subtask not found')
    task_id = r[0]
    save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time or s['completed_in_time'], completed_with_delay or s['completed_with_delay'])
    return {'message': 'updated'}


@app.delete('/backend/subtasks/{subtask_id}')
async def delete_subtask(subtask_id: str):
    s = get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('DELETE FROM subtasks WHERE id=?', (subtask_id,))
    conn.commit()
    conn.close()
    return {'message': 'deleted'}


@app.post('/backend/tasks')
async def create_task(name: str = Form(...)):
    tid = str(uuid.uuid4())
    save_task_to_db(tid, name)
    return {'task_id': tid, 'name': name}


@app.get('/backend/tasks')
async def list_tasks():
    return list_tasks_from_db()


@app.put('/backend/tasks/{task_id}')
async def update_task_endpoint(task_id: str, name: str = Form(...)):
    t = get_task_from_db(task_id)
    if not t:
        raise HTTPException(status_code=404, detail='task not found')
    save_task_to_db(task_id, name)
    return {'message': 'updated'}


@app.delete('/backend/tasks/{task_id}')
async def delete_task_endpoint(task_id: str):
    t = get_task_from_db(task_id)
    if not t:
        raise HTTPException(status_code=404, detail='task not found')
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('DELETE FROM subtasks WHERE task_id=?', (task_id,))
    cur.execute('DELETE FROM tasks WHERE id=?', (task_id,))
    conn.commit()
    conn.close()
    return {'message': 'deleted'}


@app.get('/backend/analyses')
async def list_analyses(limit: int = 100):
    try:
        return list_analyses_from_db(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/backend/analysis/{analysis_id}')
async def get_analysis(analysis_id: str):
    try:
        res = get_analysis_from_db(analysis_id)
        if res is None:
            raise HTTPException(status_code=404, detail='analysis not found')
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/backend/analysis/{analysis_id}')
async def delete_analysis(analysis_id: str):
    try:
        delete_analysis_from_db(analysis_id)
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def update_subtask_counts(subtask_id, completed_in_time_increment, completed_with_delay_increment):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('UPDATE subtasks SET completed_in_time = completed_in_time + ?, completed_with_delay = completed_with_delay + ? WHERE id = ?', (completed_in_time_increment, completed_with_delay_increment, subtask_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

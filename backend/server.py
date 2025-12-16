from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
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
import threading
import backend.rules as rules_mod

app = FastAPI()

app.state.webcam_event = threading.Event()

import backend.captioner as captioner_mod
import backend.llm as llm_mod
import backend.db as db_mod


get_local_captioner = captioner_mod.get_local_captioner
get_captioner_for_model = captioner_mod.get_captioner_for_model
_normalize_caption_output = captioner_mod._normalize_caption_output
_captioner_cache = getattr(captioner_mod, '_captioner_cache', {})
_captioner_status = getattr(captioner_mod, '_captioner_status', {})

DB_PATH = db_mod.DB_PATH
init_db = db_mod.init_db
save_analysis_to_db = db_mod.save_analysis_to_db
list_analyses_from_db = db_mod.list_analyses_from_db
get_analysis_from_db = db_mod.get_analysis_from_db
delete_analysis_from_db = db_mod.delete_analysis_from_db
save_task_to_db = db_mod.save_task_to_db
list_tasks_from_db = db_mod.list_tasks_from_db
get_task_from_db = db_mod.get_task_from_db
save_subtask_to_db = db_mod.save_subtask_to_db
list_subtasks_from_db = db_mod.list_subtasks_from_db
get_subtask_from_db = db_mod.get_subtask_from_db
compute_ranges = db_mod.compute_ranges
update_subtask_counts = db_mod.update_subtask_counts
# Timing/segmenting defaults (tunable)
MIN_SEGMENT_SEC = 0.5  # ignore segments shorter than this
MERGE_GAP_SEC = 1.0    # merge segments separated by <= this gap


def merge_and_filter_ranges(ranges, min_segment_sec=MIN_SEGMENT_SEC, merge_gap_sec=MERGE_GAP_SEC):
    """Merge nearby ranges and drop very short segments.

    ranges: list of dicts with 'startTime' and 'endTime'
    Returns a new list of merged, filtered ranges.
    """
    if not ranges:
        return []
    # Sort by start
    rs = sorted(ranges, key=lambda r: r.get('startTime', 0))
    merged = []
    cur = rs[0].copy()
    for r in rs[1:]:
        gap = r.get('startTime', 0) - cur.get('endTime', 0)
        if gap <= merge_gap_sec:
            # extend current
            cur['endTime'] = max(cur.get('endTime', 0), r.get('endTime', 0))
            cur['endFrame'] = max(cur.get('endFrame', cur.get('endFrame', 0)), r.get('endFrame', r.get('endFrame', 0)))
        else:
            # finalize current if long enough
            dur = cur.get('endTime', 0) - cur.get('startTime', 0)
            if dur >= min_segment_sec:
                merged.append(cur.copy())
            cur = r.copy()
    # last segment
    dur = cur.get('endTime', 0) - cur.get('startTime', 0)
    if dur >= min_segment_sec:
        merged.append(cur.copy())
    return merged

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

CLASSIFY_PROMPT_TEMPLATE = llm_mod.CLASSIFY_PROMPT_TEMPLATE
# DURATION_PROMPT_TEMPLATE = llm_mod.DURATION_PROMPT_TEMPLATE
TASK_COMPLETION_PROMPT_TEMPLATE = llm_mod.TASK_COMPLETION_PROMPT_TEMPLATE

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
VLM_UPLOAD_DIR = "vlm_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VLM_UPLOAD_DIR, exist_ok=True)


@app.get("/backend/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        # Choose media type by extension
        ext = os.path.splitext(filename)[1].lower()
        media_type = "application/octet-stream"
        if ext == '.mp4':
            media_type = 'video/mp4'
        elif ext == '.avi':
            media_type = 'video/x-msvideo'
        elif ext in ('.mov', '.qt'):
            media_type = 'video/quicktime'
        return FileResponse(file_path, media_type=media_type, filename=filename, headers={"Accept-Ranges": "bytes"})
    return make_alert_json('File not found', status_code=404)

# Webcam streaming state is stored in `app.state.webcam_event` (threading.Event)

@app.get("/backend/stream_pose")
async def stream_pose(model: str = Query(''), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), task_id: str = Query(None), evaluation_mode: str = Query('combined'), compare_timings: bool = Query(False), jpeg_quality: int = Query(80), max_width: Optional[int] = Query(None), save_video: bool = Query(False), rule_set: str = Query('default'), classifier: str = Query('blip_binary'), classifier_prompt: str = Query(None), classifier_mode: str = Query('binary'), sample_interval_sec: Optional[float] = Query(None)):
    """Stream webcam frames as SSE events with BLIP captioning and optional LLM classification every 2 seconds.
    Emits structured payloads similar to /vlm_local_stream, and saves analysis to DB at the end.
    """
    # Mark the webcam as active via app.state event
    app.state.webcam_event.set()
    def gen():
        # Use the event on app.state instead of a module global
        try:
            yield _sse_event({"stage": "started", "message": "live processing started"})
            
            cap = cv2.VideoCapture(0)
            captioner = get_captioner_for_model(model) if model else get_local_captioner()
            if captioner is None:
                yield _sse_event({"stage": "alert", "message": "no captioner available"})
                return
            
            current_caption = "Initializing..."
            last_inference_time = time.time()
            frame_counter = 0
            collected_samples = []
            collected_idle = []
            collected_work = []
            cumulative_work_frames = 0
            start_time = time.time()
            # Determine sampling interval for live inference (seconds)
            try:
                sample_interval = float(sample_interval_sec) if sample_interval_sec is not None and float(sample_interval_sec) > 0 else 2.0
            except Exception:
                sample_interval = 2.0
            
            writer = None
            writer_type = None
            ffmpeg_proc = None
            saved_basename = None
            saved_path = None
            record_enabled = bool(save_video)

            while app.state.webcam_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    yield _sse_event({"stage": "alert", "message": "failed to read frame"})
                    break
                frame = cv2.flip(frame, 1)
                # If recording requested, lazily create VideoWriter once we have a frame
                if record_enabled and writer is None:
                    try:
                        # Ensure processed dir exists
                        os.makedirs(PROCESSED_DIR, exist_ok=True)
                        uniq = str(uuid.uuid4())
                        saved_basename = f"live_{uniq}.mp4"
                        saved_path = os.path.join(PROCESSED_DIR, saved_basename)
                        # Determine fps (fallback to 30)
                        try:
                            cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
                        except Exception:
                            cap_fps = 30.0
                        h, w = frame.shape[:2]
                        # Prefer ffmpeg encoding (H.264) when available for browser compatibility
                        ffmpeg_path = shutil.which('ffmpeg')
                        if ffmpeg_path:
                            # Build ffmpeg command to read raw BGR frames from stdin and encode to h264 mp4
                            cmd = [
                                ffmpeg_path,
                                '-y',
                                '-f', 'rawvideo',
                                '-pix_fmt', 'bgr24',
                                '-s', f'{w}x{h}',
                                '-r', str(int(cap_fps)),
                                '-i', '-',
                                '-c:v', 'libx264',
                                '-preset', 'veryfast',
                                '-pix_fmt', 'yuv420p',
                                '-crf', '23',
                                saved_path
                            ]
                            try:
                                ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                                writer_type = 'ffmpeg'
                                writer = ffmpeg_proc
                                logging.info('Recording live stream via ffmpeg to %s (fps=%s size=%dx%d)', saved_path, cap_fps, w, h)
                            except Exception as e:
                                logging.warning('ffmpeg recording failed to start: %s', e)
                                ffmpeg_proc = None
                                writer = None
                                writer_type = None

                        # If ffmpeg unavailable or failed, fall back to OpenCV VideoWriter
                        if writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(saved_path, fourcc, cap_fps, (w, h))
                            writer_type = 'cv2'
                            # verify writer opened; if not, try AVI/XVID fallback
                            if not getattr(writer, 'isOpened', lambda: False)():
                                try:
                                    writer.release()
                                except Exception:
                                    pass
                                # fallback to AVI/XVID
                                saved_basename = f"live_{uniq}.avi"
                                saved_path = os.path.join(PROCESSED_DIR, saved_basename)
                                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                writer = cv2.VideoWriter(saved_path, fourcc, cap_fps, (w, h))
                            if not getattr(writer, 'isOpened', lambda: False)():
                                raise RuntimeError('VideoWriter failed to open for mp4 and avi fallbacks')
                            logging.info('Recording live stream to %s (fps=%s size=%dx%d)', saved_path, cap_fps, w, h)
                    except Exception as e:
                        logging.exception('Failed to create VideoWriter: %s', e)
                        # notify client and disable recording for this session
                        yield _sse_event({"stage": "alert", "message": f"Failed to create recorder: {e}"})
                        writer = None
                        record_enabled = False
                        # remove any partial file
                        try:
                            if saved_path and os.path.exists(saved_path):
                                os.remove(saved_path)
                        except Exception:
                            pass
                # If writer is active, write the raw frame (BGR)
                if writer is not None:
                    if writer_type == 'ffmpeg' and ffmpeg_proc is not None and ffmpeg_proc.stdin:
                        # Ensure contiguous bytes and write to ffmpeg stdin
                        try:
                            data = frame.tobytes()
                            ffmpeg_proc.stdin.write(data)
                        except BrokenPipeError:
                            logging.exception('ffmpeg stdin broken pipe during write')
                            # disable recording on error
                            try:
                                ffmpeg_proc.stdin.close()
                            except Exception:
                                pass
                            ffmpeg_proc = None
                            writer = None
                            writer_type = None
                    else:
                        writer.write(frame)
                frame_counter += 1
                elapsed_time = time.time() - start_time
                
                if time.time() - last_inference_time >= sample_interval:
                    try:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # Allow optional prompt to be passed into captioners that support it.
                        try:
                            out = captioner(img, prompt=prompt)
                        except TypeError:
                            out = captioner(img)
                        caption = _normalize_caption_output(captioner, out)
                        
                        # Determine label (LLM + keyword rules) via centralized helper
                        label, cls_text = rules_mod.determine_label(
                            caption,
                            use_llm=use_llm,
                            text_llm=llm_mod.get_local_text_llm(),
                            prompt=prompt,
                            classify_prompt_template=CLASSIFY_PROMPT_TEMPLATE,
                        )
                        
                        # Subtask overrun check (same as vlm_local_stream)
                        subtask_overrun = None
                        if subtask_id and label == 'work':
                            cumulative_work_frames += 1
                            try:
                                assign = get_subtask_from_db(subtask_id)
                                if assign is not None and assign.get('duration_sec') is not None:
                                    expected = assign.get('duration_sec')
                                    # derive work duration from collected_samples using compute_ranges
                                    # try:
                                    fps_assume = 30.0
                                    work_ranges = compute_ranges(collected_work, collected_samples, fps_assume)
                                    merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                    actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                    # except Exception:
                                        # actual_work_time = cumulative_work_frames / 30.0
                                    if actual_work_time > expected:
                                        subtask_overrun = True
                                    else:
                                        subtask_overrun = False
                            except Exception:
                                subtask_overrun = None
                        
                        payload = {"stage": "sample", "frame_index": frame_counter, "time_sec": elapsed_time, "caption": caption, "label": label, "llm_output": cls_text}
                        if subtask_overrun is not None:
                            payload['subtask_overrun'] = subtask_overrun
                        # Encode current frame as JPEG and include as base64 to allow clients to render a live image.
                        try:
                            enc_frame = frame
                            # Optionally downscale to max_width to reduce bandwidth
                            if max_width is not None:
                                try:
                                    h, w = enc_frame.shape[:2]
                                    if w > int(max_width):
                                        new_w = int(max_width)
                                        new_h = int(h * (new_w / w))
                                        enc_frame = cv2.resize(enc_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                except Exception as e_r:
                                    logging.debug('Failed to resize frame for SSE image: %s', e_r)
                            # Respect JPEG quality parameter
                            try:
                                ret_jpg, buf = cv2.imencode('.jpg', enc_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                            except Exception:
                                ret_jpg, buf = cv2.imencode('.jpg', enc_frame)
                            if ret_jpg and buf is not None:
                                b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                                payload['image'] = b64
                        except Exception as e:
                            logging.debug('Failed to encode frame for SSE image: %s', e)
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
            
            # release resources
            try:
                cap.release()
            except Exception:
                pass
            try:
                if writer_type == 'ffmpeg' and ffmpeg_proc is not None:
                    try:
                        ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        ffmpeg_proc.wait(timeout=10)
                    except Exception:
                        try:
                            ffmpeg_proc.kill()
                        except Exception:
                            pass
                else:
                    if writer is not None:
                        try:
                            writer.release()
                        except Exception:
                            pass
            except Exception:
                pass
            
            # Save to DB at the end (similar to vlm_local_stream)
            try:
                # Provide accurate video info if recording was saved
                vid_info = {'fps': 30.0, 'frame_count': frame_counter, 'width': None, 'height': None, 'duration': elapsed_time}
                if saved_path and os.path.exists(saved_path):
                    try:
                        # probe saved file via cv2 to get width/height/fps
                        vcap = cv2.VideoCapture(saved_path)
                        vid_fps = float(vcap.get(cv2.CAP_PROP_FPS) or 0) or vid_info['fps']
                        vid_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                        vid_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                        vcap.release()
                        vid_info['fps'] = vid_fps
                        vid_info['width'] = vid_w
                        vid_info['height'] = vid_h
                    except Exception:
                        pass
                aid = str(uuid.uuid4())
                filename_for_db = saved_basename if saved_basename else f"live_webcam_{aid}.mp4"
                video_url = f"/backend/download/{filename_for_db}" if saved_basename else None
                save_analysis_to_db(aid, filename_for_db, model or 'default', prompt, video_url, vid_info, collected_samples, subtask_id=subtask_id)
                
                # Evaluate subtasks completion from collected captions using vector store + LLM
                try:
                    captions = [s.get('caption') for s in collected_samples if s.get('caption')]
                    evals = db_mod.evaluate_subtasks_completion(captions)
                except Exception:
                    evals = []

                # Update subtask counts if compare_timings enabled
                if compare_timings and (subtask_id or task_id):
                    try:
                        # If a task_id is provided, evaluate all subtasks under that task
                        if task_id:
                            try:
                                subs = db_mod.list_subtasks_from_db()
                                subs_for_task = [s for s in subs if s.get('task_id') == task_id or s.get('task_name') == task_id]
                                for s in subs_for_task:
                                    sid = s.get('id')
                                    if evaluation_mode == 'llm_only':
                                        db_mod.aggregate_and_update_subtask(sid, collected_samples, collected_work, fps=30.0, llm=llm_mod.get_local_text_llm())
                                    else:
                                        # combined (default)
                                        db_mod.aggregate_and_update_subtask(sid, collected_samples, collected_work, fps=30.0, llm=llm_mod.get_local_text_llm())
                            except Exception:
                                logging.exception('Failed to aggregate for task_id, falling back to timing-only per-subtask')
                                # best-effort timing-only per-subtask
                                subs = db_mod.list_subtasks_from_db()
                                subs_for_task = [s for s in subs if s.get('task_id') == task_id or s.get('task_name') == task_id]
                                for s in subs_for_task:
                                    sid = s.get('id')
                                    subtask = get_subtask_from_db(sid)
                                    if subtask and collected_work:
                                        work_ranges = compute_ranges(collected_work, collected_samples, 30.0)
                                        merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                        actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                        expected_duration = subtask.get('duration_sec')
                                        if actual_work_time <= expected_duration:
                                            update_subtask_counts(sid, 1, 0)
                                        else:
                                            update_subtask_counts(sid, 0, 1)
                        else:
                            # single subtask behavior (backwards compatible)
                            try:
                                inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(subtask_id, collected_samples, collected_work, fps=30.0, llm=llm_mod.get_local_text_llm())
                                logging.info(f'Aggregate update for subtask {subtask_id}: +{inc_in} in_time, +{inc_delay} with_delay ({reason})')
                            except Exception:
                                subtask = get_subtask_from_db(subtask_id)
                                if subtask and collected_work:
                                    work_ranges = compute_ranges(collected_work, collected_samples, 30.0)
                                    merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                    actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                    expected_duration = subtask.get('duration_sec')
                                    if actual_work_time <= expected_duration:
                                        update_subtask_counts(subtask_id, 1, 0)
                                    else:
                                        update_subtask_counts(subtask_id, 0, 1)
                    except Exception as e:
                        logging.exception(f'Failed to update subtask counts for {subtask_id or task_id}')
                
                out = {"stage": "finished", "message": "live processing complete", "stored_analysis_id": aid}
                if evals:
                    out['subtask_evaluations'] = evals
                if video_url:
                    out['video_url'] = video_url
                yield _sse_event(out)
            except Exception as e:
                yield _sse_event({"stage": "alert", "message": str(e)})
        finally:
            yield _sse_event({"stage": "finished", "message": "live processing complete"})
            # Clear the event so other callers know streaming stopped
            app.state.webcam_event.clear()
    # # Add explicit CORS headers to resolve OpaqueResponseBlocking
    # headers = {
    #     "Access-Control-Allow-Origin": "*",
    #     "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    #     "Access-Control-Allow-Headers": "Content-Type",
    #     "Cache-Control": "no-cache",
    # }
    return StreamingResponse(gen(), media_type='text/event-stream')
    # return StreamingResponse(gen(), media_type='text/event-stream', headers=headers)


@app.post("/backend/stop_webcam")
async def stop_webcam():
    """Stop the webcam stream."""
    # Clear the shared event to stop any running webcam stream
    try:
        app.state.webcam_event.clear()
    except Exception:
        pass
    return {"message": "Webcam stopped"}


@app.post("/backend/upload_vlm")
async def upload_vlm(video: UploadFile = File(...)):
    """Save an uploaded video to `vlm_uploads/` and return the saved filename.
    This is used when the frontend wants to start a streaming processing session.
    """
    try:
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
    except Exception as e:
        logging.exception('Failed to save uploaded VLM video')
        return make_alert_json(f'Upload failed: {e}', status_code=500)


def _sse_event(data: dict, event: Optional[str] = None) -> bytes:
    payload = ''
    if event:
        payload += f"event: {event}\n"
    payload += "data: " + json.dumps(data, default=str) + "\n\n"
    return payload.encode('utf-8')


def make_alert_json(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={"status": "error", "alert": str(message)})


@app.get("/backend/vlm_local_stream")
async def vlm_local_stream(filename: str = Query(...), model: str = Query(...), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), task_id: str = Query(None), compare_timings: bool = Query(False), enable_mediapipe: bool = Query(False), enable_yolo: bool = Query(False), rule_set: str = Query('default'), classifier: str = Query('blip_binary'), classifier_prompt: str = Query(None), classifier_mode: str = Query('binary'), sample_interval_sec: Optional[float] = Query(None)):
    """Stream processing events (SSE) for a previously-uploaded VLM video.
    The frontend should first POST the file to `/backend/upload_vlm` and then open
    an EventSource to this endpoint with the returned `filename`.
    """
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return make_alert_json('Uploaded file not found', status_code=404)

    def gen():
        try:
            yield _sse_event({"stage": "started", "message": "processing started"})

            # Basic video info
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                yield _sse_event({"stage": "alert", "message": "failed to open video"})
                return
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / (fps if fps > 0 else 30.0)
            yield _sse_event({"stage": "video_info", "fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration})

            # Build sample indices (safe-guard)
            # If `sample_interval_sec` provided, sample frames every N seconds;
            # otherwise fall back to evenly spaced up to 30 samples.
            if sample_interval_sec is not None and sample_interval_sec > 0:
                try:
                    # include time 0 and up to duration inclusive
                    step = float(sample_interval_sec)
                    times = []
                    t = 0.0
                    while t <= duration:
                        times.append(t)
                        t += step
                    # convert times to nearest frame indices, clamp to valid range
                    indices = sorted(list({min(frame_count - 1, max(0, int(round(tt * fps)))) for tt in times}))
                    if not indices:
                        indices = [0]
                except Exception:
                    max_samples = min(30, max(1, frame_count))
                    indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))
            else:
                max_samples = min(30, max(1, frame_count))
                indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))

            captioner = get_captioner_for_model(model)
            if captioner is None:
                yield _sse_event({"stage": "alert", "message": f"no local captioner available for model '{model}'"})
                return

            # Lazy init optional detectors - NOTE: server is not configured to
            # directly import or run the standalone `mediapipe_vlm` scripts.
            # If `enable_mediapipe`/`enable_yolo` are true, we do not import
            # those script files here to avoid coupling; instead return a
            # minimal placeholder summary indicating server-side support is
            # disabled unless you explicitly enable/implement it separately.
            mp_detector = None
            yolo_detector = None
            if enable_mediapipe:
                logging.info('Mediapipe preprocessing requested but disabled in server configuration')
            if enable_yolo:
                logging.info('YOLO preprocessing requested but disabled in server configuration')

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

                    # Optional preprocessing
                    mediapipe_info = None
                    yolo_info = None
                    try:
                        if mp_detector is not None:
                            mediapipe_info = mp_detector.process_frame(frame)
                    except Exception:
                        logging.exception('MediaPipe processing failed on frame')
                    try:
                        if yolo_detector is not None:
                            yolo_info = yolo_detector.detect_objects(frame)
                    except Exception:
                        logging.exception('YOLO processing failed on frame')

                    # Allow optional prompt to be passed into captioners that support it.
                    try:
                        out = captioner(img, prompt=prompt)
                    except TypeError:
                        out = captioner(img)
                    caption = _normalize_caption_output(captioner, out)

                    # Determine label (LLM + keyword rules) via centralized helper
                    # choose classify prompt: explicit override > classifier default > global
                    classify_prompt_template = classifier_prompt or rules_mod.CLASSIFIER_PROMPTS.get(classifier)
                    label, cls_text = rules_mod.determine_label(
                        caption,
                        use_llm=use_llm,
                        text_llm=llm_mod.get_local_text_llm(),
                        prompt=prompt,
                        classify_prompt_template=classify_prompt_template or CLASSIFY_PROMPT_TEMPLATE,
                        rule_set=rule_set,
                        output_mode=classifier_mode,
                    )

                    time_sec = fi / (fps if fps > 0 else 30.0)

                    # If monitoring a subtask, compute actual work time from contiguous ranges
                    subtask_overrun = None
                    if subtask_id:
                        try:
                            assign = get_subtask_from_db(subtask_id)
                            if assign is not None and assign.get('duration_sec') is not None:
                                expected = assign.get('duration_sec')
                                # Build temporary samples/work lists that include this current sample
                                temp_sample = {'frame_index': fi, 'time_sec': time_sec, 'caption': caption, 'label': label, 'llm_output': cls_text}
                                temp_samples = (collected_samples or []) + [temp_sample]
                                temp_work = (collected_work or []) + ([fi] if label == 'work' else [])
                                work_ranges = compute_ranges(temp_work, temp_samples, fps)
                                merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                if actual_work_time > expected:
                                    subtask_overrun = True
                                else:
                                    subtask_overrun = False
                        except Exception:
                            subtask_overrun = None

                    payload = {"stage": "sample", "frame_index": fi, "time_sec": time_sec, "caption": caption, "label": label, "llm_output": cls_text}
                    if mediapipe_info is not None:
                        payload['mediapipe'] = {
                            'hand_motion_regions': mediapipe_info.get('hand_motion_regions'),
                            'is_productive_motion': mediapipe_info.get('is_productive_motion')
                        }
                    if yolo_info is not None:
                        payload['yolo'] = {
                            'objects': {k: v.get('count') for k, v in (yolo_info.get('objects') or {}).items()},
                            'object_boxes_count': len(yolo_info.get('object_boxes', []))
                        }
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
                # Evaluate subtasks completion from collected captions using vector store + LLM
                try:
                    captions = [s.get('caption') for s in collected_samples if s.get('caption')]
                    evals = db_mod.evaluate_subtasks_completion(captions)
                except Exception:
                    evals = []

                # Update subtask counts if compare_timings is enabled and subtask_id or task_id provided
                if compare_timings and (subtask_id or task_id):
                    try:
                        # If a task_id is provided, evaluate all subtasks under that task
                        if task_id:
                            try:
                                subs = db_mod.list_subtasks_from_db()
                                subs_for_task = [s for s in subs if s.get('task_id') == task_id or s.get('task_name') == task_id]
                                for s in subs_for_task:
                                    sid = s.get('id')
                                    # combined (default) or llm_only handled inside aggregate
                                    db_mod.aggregate_and_update_subtask(sid, collected_samples, collected_work, fps=fps, llm=llm_mod.get_local_text_llm())
                            except Exception:
                                logging.exception('Failed to aggregate for task_id, falling back to timing-only per-subtask')
                                subs = db_mod.list_subtasks_from_db()
                                subs_for_task = [s for s in subs if s.get('task_id') == task_id or s.get('task_name') == task_id]
                                for s in subs_for_task:
                                    sid = s.get('id')
                                    subtask = get_subtask_from_db(sid)
                                    if subtask and collected_work:
                                        work_ranges = compute_ranges(collected_work, collected_samples, fps)
                                        merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                        actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                        expected_duration = subtask.get('duration_sec')
                                        if actual_work_time <= expected_duration:
                                            update_subtask_counts(sid, 1, 0)
                                        else:
                                            update_subtask_counts(sid, 0, 1)
                        else:
                            # single subtask behavior (backwards compatible)
                            try:
                                inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(subtask_id, collected_samples, collected_work, fps=fps, llm=llm_mod.get_local_text_llm())
                                logging.info(f'Aggregate update for subtask {subtask_id}: +{inc_in} in_time, +{inc_delay} with_delay ({reason})')
                            except Exception:
                                subtask = get_subtask_from_db(subtask_id)
                                if subtask and collected_work:
                                    work_ranges = compute_ranges(collected_work, collected_samples, fps)
                                    merged_ranges = merge_and_filter_ranges(work_ranges, MIN_SEGMENT_SEC, MERGE_GAP_SEC)
                                    actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                                    expected_duration = subtask.get('duration_sec')
                                    if actual_work_time <= expected_duration:
                                        update_subtask_counts(subtask_id, 1, 0)
                                    else:
                                        update_subtask_counts(subtask_id, 0, 1)
                    except Exception as e:
                        logging.exception(f'Failed to update subtask counts for {subtask_id or task_id}')
                out = {"stage": "finished", "message": "processing complete", "video_url": f"/backend/vlm_video/{filename}", "stored_analysis_id": aid}
                if evals:
                    out['subtask_evaluations'] = evals
                yield _sse_event(out)
            except Exception:
                yield _sse_event({"stage": "finished", "message": "processing complete", "video_url": f"/backend/vlm_video/{filename}"})
        except Exception as e:
            yield _sse_event({"stage": "alert", "message": str(e)})

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.get("/backend/vlm_video/{filename}")
async def get_vlm_video(filename: str):
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return make_alert_json('File not found', status_code=404)


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
        mid = e.get('id')
        mid_l = (mid or '').lower()
        # consider model available if it's present in the captioner cache
        cached = None
        try:
            cached = _captioner_cache.get(mid) or _captioner_cache.get(mid_l) or None
        except Exception:
            cached = None
        available = bool(cached)
        # Ensure device is JSON-serializable (e.g., torch.device -> str)
        device_raw = None
        try:
            device_raw = getattr(cached, 'device', None) if cached is not None else None
        except Exception:
            device_raw = None
        device_val = None
        try:
            device_val = str(device_raw) if device_raw is not None else None
        except Exception:
            device_val = None
        models_out.append({
            "id": mid,
            "name": e.get('name'),
            "task": e.get('task', 'image-to-text'),
            "available": available,
            "probed": available,
            "device": device_val,
        })

    return {"available": False, "models": models_out, "probed": False}


@app.get('/backend/rules')
async def list_rules():
    """Return available rule sets and metadata for frontend selection."""
    try:
        out = {'rule_sets': rules_mod.list_rule_sets(), 'classifiers': rules_mod.list_classifiers()}
        return out
    except Exception as e:
        logging.exception('Failed to list rule sets: %s', e)
        return make_alert_json('Failed to list rule sets', status_code=500)


@app.get('/backend/status')
async def backend_status():
    """Return simple health/status info for frontend hints: LLM availability and VLM models."""
    try:
        llm_avail = False
        try:
            llm_avail = llm_mod.get_local_text_llm() is not None
        except Exception:
            llm_avail = False

        # reuse vlm_local_models to get model descriptors
        try:
            vlm = await vlm_local_models()
            vlm_models = vlm.get('models') if isinstance(vlm, dict) else []
        except Exception:
            vlm_models = []

        return {'llm_available': bool(llm_avail), 'vlm_models': vlm_models}
    except Exception as e:
        logging.exception('Failed to get backend status: %s', e)
        return {'llm_available': False, 'vlm_models': []}


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
            return make_alert_json('No image data provided', status_code=400)

        if not model_id:
            return make_alert_json('No model specified', status_code=400)
        
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
            return make_alert_json(f"Invalid image data: {str(e)}", status_code=400)
        
        # Optional preprocessing flags (can be supplied in JSON body)
        enable_mediapipe = bool(request.get('enable_mediapipe', False))
        enable_yolo = bool(request.get('enable_yolo', False))

        # Get the appropriate captioner
        captioner = get_captioner_for_model(model_id)
        
        if captioner is None:
            return make_alert_json(f"Model {model_id} is not available", status_code=500)
        
        # Optionally run mediapipe / yolo preprocessing: server does not
        # execute the standalone mediapipe_vlm scripts here to avoid tight
        # coupling. If the frontend requests these features, return a minimal
        # summary indicating whether the server is configured to run them.
        mediapipe_summary = None
        yolo_summary = None
        if enable_mediapipe:
            mediapipe_summary = {'available': False, 'message': 'Mediapipe preprocessing is not enabled on this server'}
        if enable_yolo:
            yolo_summary = {'available': False, 'message': 'YOLO preprocessing is not enabled on this server'}

        # Generate caption (or label via adapter)
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

            out = {
                "caption": caption_text,
                "model": model_id,
                "status": "success"
            }
            if mediapipe_summary is not None:
                out['mediapipe'] = mediapipe_summary
            if yolo_summary is not None:
                out['yolo'] = yolo_summary
            return out
        except Exception as e:
            logging.error(f"Error during caption generation with {model_id}: {e}")
            return make_alert_json(f"Caption generation failed: {str(e)}", status_code=500)
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in caption endpoint: {e}")
        return make_alert_json('Internal server error', status_code=500)


@app.post('/backend/load_vlm_model')
async def load_vlm_model(model: str = Form(...), device: str = Form(None)):
    """Load and cache a VLM/captioner model on demand.
    Returns JSON indicating whether the model was successfully loaded.
    """
    if not model:
        return {"loaded": False, "model": model, "error": "No model specified"}
    try:
        p = get_captioner_for_model(model, device_override=device)
        if p is None:
            return {"loaded": False, "model": model, "error": "failed to load model (not available)"}
        status = _captioner_status.get(model, {})
        recorded_device = status.get('device')
        recorded_reason = status.get('reason')
        device_status_message = None
        if recorded_device == 'cpu':
            device_status_message = (
                "If the backend still reports CPU the server either:\n"
                "has no CUDA-visible GPU, or\n"
                "detected insufficient free GPU memory and fell back to CPU (the loader performs a safety check)."
            )
        return {"loaded": True, "model": model, "device": recorded_device, "device_reason": recorded_reason, "device_status_message": device_status_message}
    except Exception as e:
        logging.exception('Error loading model %s', model)
        # Return structured alert JSON so frontend can show a clear message
        return make_alert_json(str(e), status_code=500)

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
    task_id = s.get('task_id')
    if not task_id:
        raise HTTPException(status_code=400, detail='subtask missing task reference')
    save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time or s.get('completed_in_time', 0), completed_with_delay or s.get('completed_with_delay', 0))
    return {'message': 'updated'}


@app.delete('/backend/subtasks/{subtask_id}')
async def delete_subtask(subtask_id: str):
    s = get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    deleted = db_mod.delete_subtask_from_db(subtask_id)
    if not deleted:
        raise HTTPException(status_code=500, detail='failed to delete subtask')
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
    deleted = db_mod.delete_task_from_db(task_id)
    if not deleted:
        raise HTTPException(status_code=500, detail='failed to delete task')
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

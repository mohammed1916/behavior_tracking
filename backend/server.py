from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import uuid
from typing import Optional
from fastapi import Query
import json
import logging
import threading
import backend.rules as rules_mod

#if env has set debug then debug mode
if os.getenv('DEBUG', '0') == '1':
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()

app = FastAPI()

app.state.webcam_event = threading.Event()

import backend.llm as llm_mod
import backend.db as db_mod


import backend.captioner as captioner_mod
from backend.stream_processor import create_stream_generator

# Timing/segmenting defaults (tunable)
MIN_SEGMENT_SEC = 0.5  # ignore segments shorter than this
MERGE_GAP_SEC = 1.0    # merge segments separated by <= this gap



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# configure logging so INFO/DEBUG messages are shown
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s - %(levelname)s %(message)s')

# write logs to a file so the frontend can tail recent activity
LOG_FILE = "server.log"
logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(LOG_FILE) for h in logger.handlers):
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s %(name)s - %(levelname)s %(message)s'))
    logger.addHandler(fh)

CLASSIFY_PROMPT_TEMPLATE = llm_mod.CLASSIFY_PROMPT_TEMPLATE
# DURATION_PROMPT_TEMPLATE = llm_mod.DURATION_PROMPT_TEMPLATE
TASK_COMPLETION_PROMPT_TEMPLATE = llm_mod.TASK_COMPLETION_PROMPT_TEMPLATE

# Create module-level logger for actual logging calls
module_logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
VLM_UPLOAD_DIR = "vlm_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VLM_UPLOAD_DIR, exist_ok=True)

# Shared helpers to keep stream endpoints consistent
@app.get("/backend/stream_pose")
async def stream_pose(model: str = Query(''), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), task_id: str = Query(None), evaluation_mode: str = Query('none'), jpeg_quality: int = Query(80), max_width: Optional[int] = Query(None), save_video: bool = Query(False), rule_set: str = Query('default'), classifier: str = Query('blip_binary'), classifier_prompt: str = Query(None), classifier_mode: str = Query('binary'), classifier_source: str = Query('llm'), sample_interval_sec: Optional[float] = Query(None), processing_mode: str = Query('current_frame'), enable_mediapipe: bool = Query(False), enable_yolo: bool = Query(False)):
    """Stream video processing with evaluation modes: 'none', 'timing_only', 'llm_only', or 'combined'.
    
    Args:
        evaluation_mode: How to evaluate subtask completion:
            - 'none': No subtask evaluation
            - 'timing_only': Compare work duration against expected duration
            - 'llm_only': Use LLM reasoning only (ignores timing)
            - 'combined': Use both LLM reasoning and timing comparison (default)
    """
    """Stream webcam frames as SSE events with BLIP captioning and optional LLM classification.
    Uses shared streaming generator that supports VLM, BoW, and LLM classifier modes.
    """
    # Mark the webcam as active
    app.state.webcam_event.set()
    
    def gen():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield _sse_event({"stage": "alert", "message": "failed to open webcam"})
            return
        
        # Use shared generator
        yield from create_stream_generator(
            video_source=cap,
            model=model,
            classifier_source=classifier_source,
            classifier_mode=classifier_mode,
            prompt=prompt,
            rule_set=rule_set,
            classifier_prompt=classifier_prompt,
            subtask_id=subtask_id,
            task_id=task_id,
            evaluation_mode=evaluation_mode,
            jpeg_quality=jpeg_quality,
            max_width=max_width,
            save_video=save_video,
            processing_mode=processing_mode,
            sample_interval_sec=sample_interval_sec,
            sse_event_fn=_sse_event,
            is_webcam=True,
            stop_event=app.state.webcam_event,
            processed_dir=PROCESSED_DIR,
            min_segment_sec=MIN_SEGMENT_SEC,
            merge_gap_sec=MERGE_GAP_SEC,
            enable_mediapipe=enable_mediapipe,
            enable_yolo=enable_yolo,
            detector_fusion_mode='cascade',  # TODO: make configurable via API
        )
        
    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post("/backend/stop_webcam")
async def stop_webcam():
    """Stop the webcam stream."""
    # Clear the shared event to stop any running webcam stream
    app.state.webcam_event.clear()
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
        print(f"Saved VLM upload to {save_path}")
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
async def vlm_local_stream(filename: str = Query(...), model: str = Query(...), prompt: str = Query(''), use_llm: bool = Query(False), subtask_id: str = Query(None), task_id: str = Query(None), evaluation_mode: str = Query('none'), enable_mediapipe: bool = Query(False), enable_yolo: bool = Query(False), rule_set: str = Query('default'), classifier: str = Query('blip_binary'), classifier_prompt: str = Query(None), classifier_mode: str = Query('binary'), classifier_source: str = Query('llm'), sample_interval_sec: Optional[float] = Query(None), processing_mode: str = Query('current_frame')):
    """Stream processing events (SSE) for a previously-uploaded VLM video.
    The frontend should first POST the file to `/backend/upload_vlm` and then open
    an EventSource to this endpoint with the returned `filename`.
    Uses shared streaming generator that supports VLM, BoW, and LLM classifier modes.
    """
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return make_alert_json('Uploaded file not found', status_code=404)

    def gen():
        try:
            # Get video metadata for frame index computation
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                yield _sse_event({"stage": "alert", "message": "failed to open video"})
                return
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            duration = frame_count / (fps if fps > 0 else 30.0)
            
            # Compute frame indices for file-mode processing
            if processing_mode == 'all_frames':
                # All frames: process every frame in the video
                indices = list(range(frame_count))
            elif processing_mode == 'fast':
                # Fast: sample ~30 frames spread evenly across video
                max_samples = min(30, max(1, frame_count))
                indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))
            elif sample_interval_sec is not None and sample_interval_sec > 0:
                # every_2s or custom interval: sample by time
                step = float(sample_interval_sec)
                times = []
                t = 0.0
                while t <= duration:
                    times.append(t)
                    t += step
                indices = sorted(list({min(frame_count - 1, max(0, int(round(tt * fps)))) for tt in times}))
                if not indices:
                    indices = [0]
            else:
                # Fallback: sample 30 frames if no mode specified
                max_samples = min(30, max(1, frame_count))
                indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))
            
            cap.release()
            
            # Use shared generator with file-mode parameters
            yield from create_stream_generator(
                video_source=cv2.VideoCapture(file_path),
                model=model,
                classifier_source=classifier_source,
                classifier_mode=classifier_mode,
                prompt=prompt,
                rule_set=rule_set,
                classifier_prompt=classifier_prompt,
                subtask_id=subtask_id,
                task_id=task_id,
                evaluation_mode=evaluation_mode,
                jpeg_quality=80,
                max_width=None,
                save_video=False,
                processing_mode=processing_mode,
                sample_interval_sec=sample_interval_sec,
                sse_event_fn=_sse_event,
                is_webcam=False,
                frame_indices=indices,
                processed_dir=PROCESSED_DIR,
                min_segment_sec=MIN_SEGMENT_SEC,
                merge_gap_sec=MERGE_GAP_SEC,
                video_url=f"/backend/vlm_video/{filename}",
                analysis_filename=filename,
                enable_mediapipe=enable_mediapipe,
                enable_yolo=enable_yolo,
                detector_fusion_mode='cascade',  # TODO: make configurable via API
            )
        except Exception as e:
            yield _sse_event({"stage": "alert", "message": str(e)})

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.get("/backend/vlm_video/{filename}")
async def get_vlm_video(filename: str):
    file_path = os.path.join(VLM_UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return make_alert_json('File not found', status_code=404)


@app.get("/backend/analysis/{analysis_id}/video_overlay")
async def get_analysis_video_overlay(
    analysis_id: str,
    show_yolo: bool = Query(True),
    show_mediapipe: bool = Query(True),
    show_info: bool = Query(True)
):
    """Generate and stream video with detector overlays.
    
    Args:
        analysis_id: Analysis ID
        show_yolo: Whether to show YOLO bounding boxes
        show_mediapipe: Whether to show MediaPipe keypoints
        show_info: Whether to show info text overlay
    
    Returns:
        Annotated video file
    """
    import tempfile
    import backend.visualization as viz_mod
    
    # Get analysis data
    analysis = db_mod.get_analysis_from_db(analysis_id)
    if not analysis:
        return make_alert_json('Analysis not found', status_code=404)
    
    video_url = analysis.get('video_url')
    if not video_url:
        return make_alert_json('No video available for this analysis', status_code=404)
    
    # Get source video path
    if video_url.startswith('/backend/vlm_video/'):
        filename = video_url.split('/')[-1]
        source_video_path = os.path.join(VLM_UPLOAD_DIR, filename)
    else:
        return make_alert_json('Unsupported video source', status_code=400)
    
    if not os.path.exists(source_video_path):
        return make_alert_json('Source video not found', status_code=404)
    
    # Check if any samples have detector metadata
    samples = analysis.get('samples', [])
    has_detector_data = any(
        s.get('detector_metadata') for s in samples
    )
    
    if not has_detector_data:
        return make_alert_json(
            'No detector data available. Enable YOLO/MediaPipe during analysis.',
            status_code=404
        )
    
    # Create temporary annotated video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        success = viz_mod.create_annotated_video(
            source_video_path,
            output_path,
            samples,
            show_yolo=show_yolo,
            show_mediapipe=show_mediapipe,
            show_info=show_info
        )
        
        if not success:
            return make_alert_json('Failed to create annotated video', status_code=500)
        
        # Stream the annotated video
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"annotated_{analysis_id}.mp4",
            background=None  # Keep file until response is sent
        )
    except Exception as e:
        module_logger.error(f"Error creating overlay video: {e}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return make_alert_json(f'Error: {str(e)}', status_code=500)


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
            cached = getattr(captioner_mod, '_captioner_cache', {}).get(mid) or getattr(captioner_mod, '_captioner_cache', {}).get(mid_l) or None
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
    """Return available label modes for frontend selection."""
    try:
        out = {'label_modes': rules_mod.list_label_modes()}
        return out
    except Exception as e:
        logging.exception('Failed to list label modes: %s', e)
        return make_alert_json('Failed to list label modes', status_code=500)


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


@app.post('/backend/load_vlm_model')
async def load_vlm_model(model: str = Form(...), device: str = Form(None)):
    """Load and cache a VLM/captioner model on demand.
    Returns JSON indicating whether the model was successfully loaded.
    """
    if not model:
        return {"loaded": False, "model": model, "error": "No model specified"}
    try:
        p = captioner_mod.get_captioner_for_model(model, device_override=device)
        if p is None:
            return {"loaded": False, "model": model, "error": "failed to load model (not available)"}
        status = getattr(captioner_mod, '_captioner_status', {}).get(model, {})
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
    s = db_mod.get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    return s


@app.get('/backend/subtasks')
async def list_subtasks():
    return db_mod.list_subtasks_from_db()


@app.post('/backend/subtasks')
async def create_subtask(
    task_id: str = Form(...),
    subtask_info: str = Form(''),
    duration_sec: int = Form(...),
):
    subtask_id = str(uuid.uuid4())
    db_mod.save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec)
    return {"subtask_id": subtask_id}


@app.put('/backend/subtasks/{subtask_id}')
async def update_subtask(subtask_id: str, subtask_info: str = Form(...), duration_sec: int = Form(...), completed_in_time: int = Form(None), completed_with_delay: int = Form(None)):
    s = db_mod.get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    task_id = s.get('task_id')
    if not task_id:
        raise HTTPException(status_code=400, detail='subtask missing task reference')
    db_mod.save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time or s.get('completed_in_time', 0), completed_with_delay or s.get('completed_with_delay', 0))
    return {'message': 'updated'}


@app.delete('/backend/subtasks/{subtask_id}')
async def delete_subtask(subtask_id: str):
    s = db_mod.get_subtask_from_db(subtask_id)
    if s is None:
        raise HTTPException(status_code=404, detail='subtask not found')
    deleted = db_mod.delete_subtask_from_db(subtask_id)
    if not deleted:
        raise HTTPException(status_code=500, detail='failed to delete subtask')
    return {'message': 'deleted'}


@app.post('/backend/tasks')
async def create_task(name: str = Form(...)):
    tid = str(uuid.uuid4())
    db_mod.save_task_to_db(tid, name)
    return {'task_id': tid, 'name': name}


@app.get('/backend/tasks')
async def list_tasks():
    return db_mod.list_tasks_from_db()


@app.put('/backend/tasks/{task_id}')
async def update_task_endpoint(task_id: str, name: str = Form(...)):
    t = db_mod.get_task_from_db(task_id)
    if not t:
        raise HTTPException(status_code=404, detail='task not found')
    db_mod.save_task_to_db(task_id, name)
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
        return db_mod.list_analyses_from_db(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/backend/analysis/{analysis_id}')
async def get_analysis(analysis_id: str):
    try:
        res = db_mod.get_analysis_from_db(analysis_id)
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
        db_mod.delete_analysis_from_db(analysis_id)
        return {"deleted": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


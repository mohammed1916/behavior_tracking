from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import uuid
from tracker import BehaviorTracker
from typing import Optional
import logging

# configure logging so INFO/DEBUG messages are shown
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Local VLM (BLIP) imports will be optional and loaded lazily
_local_captioner = None

def get_local_captioner():
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
        _local_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    except Exception:
        _local_captioner = None
    return _local_captioner

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

        # Basic video analysis via OpenCV: duration, fps, width, height
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

                # Try to read first frame for a quick visual summary
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

    # Placeholder response - in a real integration you'd call your VLM here
    return {"message": "VLM (video) stub response", "analysis": analysis}


@app.post("/backend/vlm_local")
async def vlm_local(
    model: str = Form(...),
    prompt: str = Form(''),
    video: UploadFile = File(None),
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

    # Run local captioner on multiple sampled frames if available
    captioner = get_local_captioner()
    if captioner is None:
        analysis.update({"local_captioner": "not available (install transformers/torch)"})
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
            text = None
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                if isinstance(first, dict):
                    text = first.get('generated_text') or first.get('caption') or str(first)
                else:
                    text = str(first)
            else:
                text = str(out)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

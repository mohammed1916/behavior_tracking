"""Shared streaming generator for both webcam and file-based video processing.

This module extracts common logic from /backend/stream_pose and /backend/vlm_local_stream
to avoid code duplication while supporting all three classifier modes (VLM, BoW, LLM).
"""

import os
import time
import uuid
import logging
import cv2
from typing import Optional, Generator, Any, Dict, List
from PIL import Image

import backend.db as db_mod
import backend.llm as llm_mod
import backend.rules as rules_mod
import backend.evidence as evidence_mod
import backend.captioner as captioner_mod
from backend.stream_utils import (
    normalize_caption_output,
    call_captioner,
    build_vlm_prompt_for_source,
    build_classify_prompt_template,
    per_sample_label_for_source,
    merge_and_filter_ranges,
    compute_sample_interval,
    encode_frame_for_sse_image,
    probe_saved_video_info,
    start_live_recording,
)

logger = logging.getLogger(__name__)


# ============================================================================
# MODULAR COMPONENTS: VLM Inference, Timeline, and Activity Segmentation
# ============================================================================

def get_frame_caption(frame, captioner, vlm_prompt: str) -> str:
    """Extract caption from a single frame using VLM.
    
    Args:
        frame: OpenCV frame (BGR numpy array)
        captioner: Captioner instance
        vlm_prompt: Prompt for the VLM
    
    Returns:
        Normalized caption string
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out = call_captioner(captioner, img, vlm_prompt, on_debug=None)
    caption = normalize_caption_output(captioner, out)
    return caption


class CaptionTimeline:
    """Stores timestamped captions and provides text formatting for LLM consumption."""
    
    def __init__(self, start_time: float, end_time: float):
        self.start_time = start_time
        self.end_time = end_time
        self.samples: List[Dict[str, Any]] = []
    
    def add_sample(self, time_sec: float, caption: str) -> None:
        """Add a timestamped caption sample."""
        self.samples.append({'t': time_sec, 'caption': caption})
        self.end_time = time_sec
    
    def to_timeline_text(self) -> str:
        """Convert samples to formatted timeline text for LLM."""
        lines = []
        for sample in self.samples:
            lines.append(f"t={sample['t']:.1f}s: {sample['caption']}")
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.samples)


def segment_activities(
    timeline_text: str,
    llm_prompt_template: str,
    user_prompt: str,
    classifier_mode: str,
    caption_samples: List[Dict[str, Any]],
    min_samples: int = 1
) -> List[Dict[str, Any]]:
    """Segment activities from timeline text using LLM reasoning.
    
    Args:
        timeline_text: Formatted timeline of captions
        llm_prompt_template: Template for classification prompt
        user_prompt: User's custom prompt
        classifier_mode: 'binary' or 'multi'
        caption_samples: List of {'t': float, 'caption': str} for parsing
        min_samples: Minimum samples required
    
    Returns:
        List of segment dictionaries with start_time, end_time, label, etc.
    """
    if not timeline_text or len(caption_samples) < min_samples:
        return []
    
    rendered = llm_prompt_template.format(caption=timeline_text, prompt=user_prompt)
    
    try:
        text_llm = llm_mod.get_local_text_llm()
        llm_res = text_llm(rendered, max_new_tokens=100)
        
        if isinstance(llm_res, list) and llm_res and isinstance(llm_res[0], dict):
            llm_output = llm_res[0].get('generated_text', str(llm_res))
        else:
            llm_output = str(llm_res)
        
        logger.info(f"[LLM_INPUT] captions: {[s['caption'][:40] for s in caption_samples]}")
        logger.info(f"[LLM_OUTPUT] {llm_output.replace(chr(10), ' ')[:200]}")
        
        # Import here to avoid circular dependency
        from backend.stream_utils import parse_llm_segments
        
        segments = parse_llm_segments(llm_output, caption_samples, classifier_mode, prompt=rendered)
        
        for seg in segments:
            seg['llm_output'] = llm_output
        
        return segments
    
    except Exception as e:
        logger.exception('Failed to segment activities: %s', e)
        return []


def create_stream_generator(
    video_source: cv2.VideoCapture,
    model: str,
    classifier_source: str,
    classifier_mode: str,
    prompt: str,
    rule_set: str,
    classifier_prompt: Optional[str],
    subtask_id: Optional[str],
    task_id: Optional[str],
    compare_timings: bool,
    evaluation_mode: str,
    jpeg_quality: int,
    max_width: Optional[int],
    save_video: bool,
    processing_mode: str,
    sample_interval_sec: Optional[float],
    sse_event_fn,  # Function to emit SSE events
    is_webcam: bool = False,
    frame_indices: Optional[List[int]] = None,
    stop_event = None,  # For webcam: app.state.webcam_event
    processed_dir: str = "processed",
    min_segment_sec: float = 0.5,
    merge_gap_sec: float = 1.0,
    fps_override: Optional[float] = None,
    duration_override: Optional[float] = None,
    video_url: Optional[str] = None,  # For file mode: e.g. /backend/vlm_video/filename
    analysis_filename: Optional[str] = None,  # For file mode: uploaded filename
) -> Generator[bytes, None, None]:
    """
    Shared streaming generator for webcam and file-based processing.
    
    Args:
        video_source: Opened cv2.VideoCapture object
        model: VLM model identifier
        classifier_source: 'vlm', 'bow', or 'llm'
        classifier_mode: 'binary' or 'multi'
        prompt: User-provided custom prompt
        rule_set: Rule set identifier for BoW mode
        classifier_prompt: Custom classifier prompt
        subtask_id: Optional subtask ID for monitoring
        task_id: Optional task ID (comma-separated list)
        compare_timings: Whether to evaluate subtask timing
        evaluation_mode: 'combined' or 'llm_only'
        jpeg_quality: JPEG encoding quality (1-100)
        max_width: Max width for encoded frames (None=no limit)
        save_video: Whether to record video (webcam only)
        processing_mode: 'current_frame', 'every_2s'
        sample_interval_sec: Custom sampling interval
        sse_event_fn: Function(dict) -> bytes to emit SSE events
        is_webcam: True for live webcam, False for uploaded file
        frame_indices: List of frame indices (for file mode)
        stop_event: Threading event to check if should stop (webcam only)
        processed_dir: Directory to save recordings
        min_segment_sec: Min segment duration to keep
        merge_gap_sec: Max gap to merge segments
        fps_override: Override FPS (for pre-computed values)
        duration_override: Override duration (for pre-computed values)
        video_url: URL to serve the video (file mode only, e.g., /backend/vlm_video/filename)
        analysis_filename: Original filename for database (file mode only)
    
    Yields:
        SSE event bytes for: started, video_info, samples, segments, finished, etc.
    """
    
    # Normalize classifier source
    classifier_source_norm = (classifier_source or 'llm').lower()
    effective_use_llm = (classifier_source_norm == 'llm')
    
    # Initialize tracking variables
    frame_counter = 0
    collected_samples = []
    collected_idle = []
    collected_work = []
    collected_segments = []
    cumulative_work_frames = 0
    start_time = time.time()
    last_inference_time = time.time()
    
    # Window aggregation (LLM mode only)
    current_window = None
    max_samples_per_window = 4
    min_samples = 1
    seen_seg_keys = set()
    
    # Pre-build VLM prompt and classification template (used for all frames)
    vlm_prompt = build_vlm_prompt_for_source(classifier_source_norm, classifier_mode, classifier_prompt)
    classify_prompt_template = build_classify_prompt_template(classifier_source_norm, classifier_mode, classifier_prompt)
    
    # Recording state (webcam only)
    writer = None
    writer_type = None
    ffmpeg_proc = None
    saved_basename = None
    saved_path = None
    
    try:
        yield sse_event_fn({"stage": "started", "message": "processing started"})
        
        # Get captioner
        captioner = captioner_mod.get_captioner_for_model(model) if model else captioner_mod.get_local_captioner()
        if captioner is None:
            yield sse_event_fn({"stage": "alert", "message": "no captioner available"})
            return
        
        # Get video metadata
        frame_count = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = fps_override or float(video_source.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = duration_override or (frame_count / (fps if fps > 0 else 30.0))
        
        if not is_webcam:
            # For file mode, emit video_info
            yield sse_event_fn({"stage": "video_info", "fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration})
        
        # Determine sampling
        if is_webcam:
            # Webcam: continuous sampling with interval
            sample_interval = compute_sample_interval(processing_mode, sample_interval_sec)
            frame_source = _webcam_frame_generator(
                video_source, 
                sample_interval, 
                stop_event,
                processed_dir,
                save_video,
            )
        else:
            # File: sample specific frames
            sample_interval = None
            if frame_indices is None:
                # Auto-generate indices if not provided
                if sample_interval_sec is not None and sample_interval_sec > 0:
                    step = 2.0 if processing_mode == 'every_2s' else float(sample_interval_sec)
                    times = []
                    t = 0.0
                    while t <= duration:
                        times.append(t)
                        t += step
                    frame_indices = sorted(list({min(frame_count - 1, max(0, int(round(tt * fps)))) for tt in times}))
                    if not frame_indices:
                        frame_indices = [0]
                else:
                    max_samples = min(30, max(1, frame_count))
                    frame_indices = sorted(list({int(i * frame_count / max_samples) for i in range(max_samples)}))
            
            frame_source = _file_frame_generator(video_source, frame_indices)
        
        # Process frames
        for frame_data in frame_source:
            if is_webcam:
                frame, elapsed_time, writer, writer_type, ffmpeg_proc, saved_basename, saved_path = frame_data
                frame_counter += 1
            else:
                frame, fi = frame_data
                elapsed_time = fi / (fps if fps > 0 else 30.0)
                frame_counter = fi
            
            # Skip inference if interval not met (webcam mode)
            if is_webcam and (time.time() - last_inference_time) < sample_interval:
                continue
            
            try:
                # VLM inference: single call per frame
                caption = get_frame_caption(frame, captioner, vlm_prompt)
                
                # Text-based classification: caption → label
                label, cls_text = per_sample_label_for_source(
                    classifier_source_norm,
                    classifier_mode,
                    caption,
                    prompt,
                    rule_set,
                    classify_prompt_template,
                    effective_use_llm,
                )
                
                # Check subtask overrun
                subtask_overrun = None
                if subtask_id and label == 'work':
                    cumulative_work_frames += 1
                    try:
                        assign = db_mod.get_subtask_from_db(subtask_id)
                        if assign is not None and assign.get('duration_sec') is not None:
                            expected = assign.get('duration_sec')
                            work_ranges = db_mod.compute_ranges(collected_work, collected_samples, fps)
                            merged_ranges = merge_and_filter_ranges(work_ranges, min_segment_sec, merge_gap_sec)
                            actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                            subtask_overrun = actual_work_time > expected
                    except Exception:
                        subtask_overrun = None
                
                # Emit sample
                payload = {
                    "stage": "sample",
                    "frame_index": frame_counter,
                    "time_sec": elapsed_time,
                    "caption": caption,
                    "label": label,
                    "llm_output": cls_text,
                }
                if subtask_overrun is not None:
                    payload['subtask_overrun'] = subtask_overrun
                
                # Encode image if requested
                if is_webcam and jpeg_quality:
                    try:
                        b64 = encode_frame_for_sse_image(frame, jpeg_quality=jpeg_quality, max_width=max_width)
                        if b64:
                            payload['image'] = b64
                    except Exception:
                        pass
                
                yield sse_event_fn(payload)
                
                # Collect sample
                sample = {
                    'frame_index': frame_counter,
                    'time_sec': elapsed_time,
                    'caption': caption,
                    'label': label,
                    'llm_output': cls_text,
                }
                collected_samples.append(sample)
                if label == 'work':
                    collected_work.append(frame_counter)
                else:
                    collected_idle.append(frame_counter)
                
                # Window aggregation (LLM mode only)
                if classifier_source_norm == 'llm':
                    print(f"Adding sample to window at t={elapsed_time:.2f}s: {caption[:50]}...")
                    if current_window is None:
                        current_window = CaptionTimeline(
                            start_time=elapsed_time,
                            end_time=elapsed_time,
                        )
                    
                    current_window.add_sample(elapsed_time, caption)
                    
                    # Check window closure
                    if len(current_window) >= max_samples_per_window:
                        timeline_text = current_window.to_timeline_text()
                        segments = segment_activities(
                            timeline_text,
                            classify_prompt_template,
                            prompt,
                            classifier_mode,
                            current_window.samples,
                            min_samples,
                        )
                        
                        for seg_event in segments:
                            collected_segments.append(seg_event)
                            yield sse_event_fn(seg_event)
                            
                            # Update samples in segment range
                            segment_label = seg_event.get('label')
                            segment_start = seg_event.get('start_time', 0)
                            segment_end = seg_event.get('end_time', 0)
                            
                            for sample in collected_samples:
                                sample_time = sample.get('time_sec', 0)
                                if segment_start <= sample_time <= segment_end:
                                    if segment_label and sample.get('label') is None:
                                        sample['label'] = segment_label
                                        if segment_label == 'idle':
                                            collected_idle.append(sample['frame_index'])
                                        else:
                                            collected_work.append(sample['frame_index'])
                        
                        current_window = None
                
                last_inference_time = time.time()
                
            except Exception as e:
                yield sse_event_fn({"stage": "sample_error", "frame_index": frame_counter, "error": str(e)})
        
        # Flush remaining window
        if classifier_source_norm == 'llm' and current_window is not None:
            timeline_text = current_window.to_timeline_text()
            segments = segment_activities(
                timeline_text,
                classify_prompt_template,
                prompt,
                classifier_mode,
                current_window.samples,
                min_samples,
            )
            for seg_event in segments:
                k = (round(float(seg_event.get('start_time', 0)), 2),
                     round(float(seg_event.get('end_time', 0)), 2),
                     str(seg_event.get('label') or '').lower())
                if k not in seen_seg_keys:
                    collected_segments.append(seg_event)
                    seen_seg_keys.add(k)
                    yield sse_event_fn(seg_event)
        
        # Cleanup
        if is_webcam:
            _cleanup_recording(writer, writer_type, ffmpeg_proc)
        
        # Save to DB
        try:
            vid_info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
            }
            
            # For webcam recordings, probe the actual file
            if is_webcam and saved_path and os.path.exists(saved_path):
                vid_info, err = probe_saved_video_info(saved_path, fps, frame_count, duration)
            
            aid = str(uuid.uuid4())
            
            if is_webcam:
                filename_for_db = saved_basename or f"live_webcam_{aid}.mp4"
                final_video_url = f"/backend/download/{filename_for_db}" if saved_basename else None
            else:
                # File mode: use provided analysis_filename and video_url
                filename_for_db = analysis_filename or f"vlm_upload_{aid}.mp4"
                final_video_url = video_url
            
            db_mod.save_analysis_to_db(
                aid, filename_for_db, model or 'default', prompt, final_video_url,
                vid_info, collected_samples, subtask_id=subtask_id, segments=collected_segments,
            )
            
            # Evaluate subtasks
            try:
                captions = [s.get('caption') for s in collected_samples if s.get('caption')]
                evals = db_mod.evaluate_subtasks_completion(captions)
            except Exception:
                evals = []
            
            # Update subtask counts
            if compare_timings and (subtask_id or task_id):
                _update_subtask_counts(
                    subtask_id, task_id, collected_samples, collected_work, fps,
                    evaluation_mode, min_segment_sec, merge_gap_sec,
                )
            
            out = {
                "stage": "finished",
                "message": "processing complete",
                "stored_analysis_id": aid,
            }
            if evals:
                out['subtask_evaluations'] = evals
            if final_video_url:
                out['video_url'] = final_video_url
            
            yield sse_event_fn(out)
            
        except Exception as e:
            yield sse_event_fn({"stage": "alert", "message": str(e)})
    
    except Exception as e:
        yield sse_event_fn({"stage": "alert", "message": str(e)})
    
    finally:
        yield sse_event_fn({"stage": "finished", "message": "processing complete"})
        video_source.release()


def _webcam_frame_generator(cap, sample_interval, stop_event, processed_dir, save_video):
    """Generator for webcam frames with recording."""
    writer = None
    writer_type = None
    ffmpeg_proc = None
    saved_basename = None
    saved_path = None
    start_time = time.time()
    
    try:
        while stop_event is None or stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            elapsed_time = time.time() - start_time
            
            # Handle recording
            if save_video and writer is None:
                try:
                    rec_gen = start_live_recording(frame, cap, processed_dir)
                    try:
                        state = rec_gen.send(None)
                    except StopIteration as si:
                        state = si.value
                    writer = state['writer']
                    writer_type = state['writer_type']
                    ffmpeg_proc = state['ffmpeg_proc']
                    saved_basename = state['saved_basename']
                    saved_path = state['saved_path']
                except Exception:
                    writer = None
            
            if writer is not None:
                if writer_type == 'ffmpeg' and ffmpeg_proc is not None and ffmpeg_proc.stdin:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        writer = None
                        ffmpeg_proc = None
                else:
                    try:
                        writer.write(frame)
                    except Exception:
                        pass
            
            yield frame, elapsed_time, writer, writer_type, ffmpeg_proc, saved_basename, saved_path
    
    finally:
        _cleanup_recording(writer, writer_type, ffmpeg_proc)


def _file_frame_generator(cap, frame_indices):
    """Generator for file frames at specific indices."""
    for fi in frame_indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                yield frame, fi
        except Exception:
            pass





def _cleanup_recording(writer, writer_type, ffmpeg_proc):
    """Clean up video writer resources."""
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
        elif writer is not None:
            try:
                writer.release()
            except Exception:
                pass
    except Exception:
        pass


def _update_subtask_counts(subtask_id, task_id, collected_samples, collected_work, fps,
                           evaluation_mode, min_segment_sec, merge_gap_sec):
    """Update subtask completion counts."""
    try:
        if task_id:
            subs = db_mod.list_subtasks_from_db()
            task_ids = [t.strip() for t in (task_id or '').split(',') if t.strip()]
            subs_for_task = [s for s in subs if (s.get('task_id') in task_ids) or (s.get('task_name') in task_ids)]
            
            for s in subs_for_task:
                sid = s.get('id')
                try:
                    db_mod.aggregate_and_update_subtask(
                        sid, collected_samples, collected_work, fps=fps,
                        llm=llm_mod.get_local_text_llm(),
                    )
                except Exception:
                    # Fallback: timing-only
                    subtask = db_mod.get_subtask_from_db(sid)
                    if subtask and collected_work:
                        work_ranges = db_mod.compute_ranges(collected_work, collected_samples, fps)
                        merged_ranges = merge_and_filter_ranges(work_ranges, min_segment_sec, merge_gap_sec)
                        actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                        expected = subtask.get('duration_sec')
                        if actual_work_time <= expected:
                            db_mod.update_subtask_counts(sid, 1, 0)
                        else:
                            db_mod.update_subtask_counts(sid, 0, 1)
        
        elif subtask_id:
            try:
                db_mod.aggregate_and_update_subtask(
                    subtask_id, collected_samples, collected_work, fps=fps,
                    llm=llm_mod.get_local_text_llm(),
                )
            except Exception:
                subtask = db_mod.get_subtask_from_db(subtask_id)
                if subtask and collected_work:
                    work_ranges = db_mod.compute_ranges(collected_work, collected_samples, fps)
                    merged_ranges = merge_and_filter_ranges(work_ranges, min_segment_sec, merge_gap_sec)
                    actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged_ranges)
                    expected = subtask.get('duration_sec')
                    if actual_work_time <= expected:
                        db_mod.update_subtask_counts(subtask_id, 1, 0)
                    else:
                        db_mod.update_subtask_counts(subtask_id, 0, 1)
    
    except Exception as e:
        logger.exception('Failed to update subtask counts: %s', e)

import base64
from typing import Optional, List, Dict, Any, Callable, Tuple
import cv2
import os
import uuid
import subprocess
import shutil
import logging
import inspect
import re

import backend.llm as llm_mod
import backend.rules as rules_mod


def parse_llm_segments(llm_output: str, all_captions: List[Dict], classifier_mode: str) -> List[Dict]:
    """Parse LLM timeline segmentation output into structured segments.
    
    Expected format from LLM:
    0.50-2.30: work
    3.10-3.10: using_phone
    3.80-5.20: work
    
    Returns list of segments with start_time, end_time, label, captions.
    """
    segments = []
    
    # Extract only clean segment lines (ignore any preamble like "Thinking...")
    # Split output by newlines and process only lines with the segment format
    segment_lines = []
    for line in llm_output.split('\n'):
        line = line.strip()
        # Skip empty lines and lines that don't look like segments
        if not line or ':' not in line:
            continue
        # Check if line matches pattern: [number]-[number]: [label]
        if re.match(r'^\d+\.?\d*\s*-\s*\d+\.?\d*\s*:\s*\w+$', line):
            segment_lines.append(line)
    
    matches = []
    pattern = r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*:\s*(\w+)'
    for line in segment_lines:
        m = re.match(pattern, line)
        if m:
            matches.append(m.groups())
    
    if not matches:
        # Fallback: if LLM output is malformed, create single segment with normalized label
        label = rules_mod.normalize_label_text(llm_output, output_mode=classifier_mode)
        if all_captions:
            return [{
                'stage': 'segment',
                'start_time': all_captions[0]['t'],
                'end_time': all_captions[-1]['t'],
                'label': label,
                'captions': [c['caption'] for c in all_captions],
                'timestamps': [c['t'] for c in all_captions],
                'duration': all_captions[-1]['t'] - all_captions[0]['t'],
                'timeline': '\n'.join([f"<t={c['t']:.2f}> {c['caption']}" for c in all_captions])
            }]
        return []
    
    # Process each segment, skip invalid ranges
    for start_str, end_str, label_str in matches:
        try:
            start_time = float(start_str)
            end_time = float(end_str)
        except (ValueError, TypeError) as e:
            print(f'Failed to parse segment timestamps: {start_str}-{end_str}: {e}')
            continue
        
        # Skip invalid ranges: start must be less than end (reject start >= end)
        if start_time >= end_time:
            print(f'Skipping invalid segment: {start_time}-{end_time} (start >= end)')
            continue
        
        label = rules_mod.normalize_label_text(label_str, output_mode=classifier_mode)
        
        # Find captions within this time range (with small tolerance)
        tolerance = 0.05  # 50ms tolerance for floating point comparison
        segment_captions = [
            c for c in all_captions 
            if start_time <= c['t'] <= end_time
        ]
        
        if segment_captions:
            segments.append({
                'stage': 'segment',
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'captions': [c['caption'] for c in segment_captions],
                'timestamps': [c['t'] for c in segment_captions],
                'duration': end_time - start_time,
                'timeline': '\n'.join([f"<t={c['t']:.2f}> {c['caption']}" for c in segment_captions])
            })
    
    # Remove duplicate/overlapping segments (LLM sometimes outputs overlapping ranges)
    # Keep segment with earlier start time or (if same start) earlier end time
    deduplicated = []
    seen_ranges = set()
    
    for seg in segments:
        start = seg['start_time']
        end = seg['end_time']
        # Use a tuple key for exact time matching (with small tolerance)
        time_key = (round(start, 2), round(end, 2))
        if time_key not in seen_ranges:
            deduplicated.append(seg)
            seen_ranges.add(time_key)
        else:
            print(f'Skipping duplicate segment: {start}-{end}:{seg["label"]} (already seen)')
    
    return deduplicated


def merge_segments_with_pending(
    new_segments: List[Dict[str, Any]],
    pending_segment: Optional[Dict[str, Any]],
    *,
    merge_gap_sec: float = 0.2,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Merge the head of `new_segments` with a carried pending segment.

    Strategy: keep the tail segment un-emitted (pending) so the next window
    can extend it; emit all other segments immediately.
    """
    emitted: List[Dict[str, Any]] = []
    segs = list(new_segments)

    # Pre-merge pending with first new segment if label matches and gap small
    if pending_segment and segs:
        head = segs[0]
        gap = head['start_time'] - pending_segment['end_time']
        if pending_segment.get('label') == head.get('label') and gap <= merge_gap_sec:
            pending_segment['end_time'] = head['end_time']
            pending_segment['duration'] = pending_segment['end_time'] - pending_segment['start_time']
            pending_segment['captions'] = pending_segment.get('captions', []) + head.get('captions', [])
            pending_segment['timestamps'] = pending_segment.get('timestamps', []) + head.get('timestamps', [])
            pending_segment['timeline'] = pending_segment.get('timeline', '') + ("\n" if pending_segment.get('timeline') else "") + head.get('timeline', '')
            pending_segment['llm_output'] = head.get('llm_output') or pending_segment.get('llm_output')
            segs = segs[1:]
        else:
            emitted.append(pending_segment)
            pending_segment = None

    # Emit all but the last segment; keep last as pending for possible merge with next window
    if segs:
        for seg in segs[:-1]:
            emitted.append(seg)
        pending_segment = segs[-1]

    return emitted, pending_segment


def normalize_caption_output(captioner, output: Any) -> str:
    """Normalize captioner output to a single caption string.

    Handles various output formats:
    - [{"generated_text": "..."}] (Qwen, transformers pipeline)
    - "plain string"
    - {"caption": "..."}
    """
    import json
    
    if not output:
        return ""

    # Handle list format (common from transformers pipelines)
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            text = first.get('generated_text') or first.get('caption') or ""
            return text.strip() if text else ""
        return str(first).strip()

    # Handle dict format
    if isinstance(output, dict):
        text = output.get('generated_text') or output.get('caption') or ""
        return text.strip() if text else ""

    # Handle string format (including stringified dicts)
    output_str = str(output).strip()
    
    # Try to parse if it looks like a stringified dict
    if output_str.startswith('{') and output_str.endswith('}'):
        try:
            parsed = json.loads(output_str)
            if isinstance(parsed, dict):
                text = parsed.get('generated_text') or parsed.get('caption') or ""
                return text.strip() if text else ""
        except (json.JSONDecodeError, ValueError):
            pass

    return output_str


def call_captioner(captioner, img, prompt: Optional[str], on_debug: Optional[Callable[[str], None]] = None):
    """Call captioner with optional prompt and debug callback, passing only supported kwargs.

    Uses inspect.signature to detect accepted parameters to avoid relying on nested TypeError fallbacks.
    """
    eff_prompt = None if (prompt is None or str(prompt).strip() == "") else prompt

    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(captioner)
        params = sig.parameters
    except (TypeError, ValueError):
        # If we cannot introspect, attempt a simple direct call as fallback
        params = {}

    if eff_prompt is not None and 'prompt' in params:
        kwargs['prompt'] = eff_prompt
    if on_debug is not None and 'on_debug' in params:
        kwargs['on_debug'] = on_debug

    try:
        return captioner(img, **kwargs)
    except TypeError:
        # Final fallback in case of unexpected signature quirks
        return captioner(img)


def build_vlm_prompt_for_source(classifier_source_norm: str, classifier_mode: str, classifier_prompt: Optional[str]) -> Optional[str]:
    """Return the prompt sent to the VLM for the given classifier source.
    
    Returns a non-empty prompt for image captioning/description, or None to use model defaults.
    """
    if classifier_source_norm == 'vlm':
        label_tpl = llm_mod.LABLE_PROMPT_TEMPLATE_MULTI if classifier_mode == 'multi' else llm_mod.LABLE_PROMPT_TEMPLATE_BINARY
        prompt = classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + label_tpl + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
        return prompt if prompt and prompt.strip() else None
    
    if classifier_source_norm == 'bow':
        prompt = classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
        return prompt if prompt and prompt.strip() else None
    
    # 'llm' mode: ask for plain description; let model use its base template
    if classifier_prompt:
        return classifier_prompt if classifier_prompt.strip() else None
    
    # Return a simple description task prompt
    return llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\nDescribe the visible activities and actions of the main person in detail."


def build_classify_prompt_template(classifier_source_norm: str, classifier_mode: str, classifier_prompt: Optional[str]) -> Optional[str]:
    """Return classify prompt template used for downstream LLM label decisions."""
    if classifier_source_norm == 'vlm':
        label_tpl = llm_mod.LABLE_PROMPT_TEMPLATE_MULTI if classifier_mode == 'multi' else llm_mod.LABLE_PROMPT_TEMPLATE_BINARY
        return classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + label_tpl + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
    if classifier_source_norm == 'bow':
        return None
    # 'llm' mode: use timeline segmentation prompts that analyze temporal patterns
    if classifier_source_norm == 'llm':
        if classifier_mode == 'multi':
            return classifier_prompt or llm_mod.LLM_SEGMENT_TIMELINE_MULTI
        else:  # binary
            return classifier_prompt or llm_mod.LLM_SEGMENT_TIMELINE_BINARY
    return classifier_prompt or rules_mod.get_label_template(classifier_mode)


def per_sample_label_for_source(
    classifier_source_norm: str,
    classifier_mode: str,
    caption: str,
    prompt: str,
    rule_set: str,
    classify_prompt_template: Optional[str],
    effective_use_llm: bool,
):
    """Compute per-sample label + text. For 'llm' source we defer labeling to windowed LLM (returns None)."""
    if classifier_source_norm == 'vlm':
        label = rules_mod.normalize_label_text(caption, output_mode=classifier_mode)
        return label, caption
    if classifier_source_norm == 'bow':
        output_mode = 'multi' if classifier_mode == 'label' else classifier_mode
        label, cls_text = rules_mod.determine_label(
            caption,
            use_llm=False,
            text_llm=None,
            prompt=prompt,
            classify_prompt_template=classify_prompt_template,
            rule_set=rule_set,
            output_mode=output_mode,
        )
        return label, cls_text
    return None, None


def merge_and_filter_ranges(ranges: List[Dict[str, Any]], min_segment_sec: float, merge_gap_sec: float) -> List[Dict[str, Any]]:
    """Merge nearby ranges and drop very short segments.

    ranges: list of dicts with 'startTime' and 'endTime'
    Returns a new list of merged, filtered ranges.
    """
    if not ranges:
        return []
    rs = sorted(ranges, key=lambda r: r.get('startTime', 0))
    merged: List[Dict[str, Any]] = []
    cur = rs[0].copy()
    for r in rs[1:]:
        gap = r.get('startTime', 0) - cur.get('endTime', 0)
        if gap <= merge_gap_sec:
            cur['endTime'] = max(cur.get('endTime', 0), r.get('endTime', 0))
            cur['endFrame'] = max(cur.get('endFrame', cur.get('endFrame', 0)), r.get('endFrame', r.get('endFrame', 0)))
        else:
            dur = cur.get('endTime', 0) - cur.get('startTime', 0)
            if dur >= min_segment_sec:
                merged.append(cur.copy())
            cur = r.copy()
    dur = cur.get('endTime', 0) - cur.get('startTime', 0)
    if dur >= min_segment_sec:
        merged.append(cur.copy())
    return merged


def compute_sample_interval(processing_mode: str, sample_interval_sec: Optional[float]) -> float:
    """Compute sampling interval in seconds given processing mode and optional param."""
    if (processing_mode or '').strip().lower() == 'every_2s':
        return 2.0
    try:
        if sample_interval_sec is not None:
            v = float(sample_interval_sec)
            if v > 0:
                return v
    except Exception:
        pass
    return 2.0


def encode_frame_for_sse_image(frame, jpeg_quality: int = 80, max_width: Optional[int] = None) -> Optional[str]:
    """Resize (optional) and JPEG-encode a BGR frame, returning base64 string or None."""
    try:
        enc_frame = frame
        if max_width is not None:
            try:
                h, w = enc_frame.shape[:2]
                if w > int(max_width):
                    new_w = int(max_width)
                    new_h = int(h * (new_w / w))
                    enc_frame = cv2.resize(enc_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        try:
            ret_jpg, buf = cv2.imencode('.jpg', enc_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        except Exception:
            ret_jpg, buf = cv2.imencode('.jpg', enc_frame)
        if ret_jpg and buf is not None:
            return base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        return None
    return None


def probe_saved_video_info(saved_path: str, default_fps: float, frame_count: int, duration: float) -> (Dict[str, Any], Optional[str]):
    """Probe saved video file for metadata via OpenCV. Returns (vid_info, error_msg)."""
    vid_info = {'fps': default_fps, 'frame_count': frame_count, 'width': None, 'height': None, 'duration': duration}
    err = None
    try:
        vcap = cv2.VideoCapture(saved_path)
        vid_fps = float(vcap.get(cv2.CAP_PROP_FPS) or 0) or default_fps
        vid_w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        vid_h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        vcap.release()
        vid_info['fps'] = vid_fps
        vid_info['width'] = vid_w
        vid_info['height'] = vid_h
    except Exception as e:
        err = str(e)
    return vid_info, err


def start_live_recording(frame, cap, processed_dir: str):
    """Generator that sets up live recording with ffmpeg/OpenCV fallbacks.
    
    Yields SSE debug event dicts (stage='debug', message=...).
    Returns final state dict: {writer, writer_type, ffmpeg_proc, saved_basename, saved_path}.
    """
    os.makedirs(processed_dir, exist_ok=True)
    uniq = str(uuid.uuid4())
    saved_basename = f"live_{uniq}.mp4"
    saved_path = os.path.join(processed_dir, saved_basename)
    
    try:
        cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    except Exception:
        cap_fps = 30.0
    
    h, w = frame.shape[:2]
    writer = None
    writer_type = None
    ffmpeg_proc = None
    
    # Try ffmpeg first
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        cmd = [
            ffmpeg_path, '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}', '-r', str(int(cap_fps)), '-i', '-',
            '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-crf', '23', saved_path
        ]
        try:
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            writer_type = 'ffmpeg'
            writer = ffmpeg_proc
            print('Recording live stream via ffmpeg to %s (fps=%s size=%dx%d)', saved_path, cap_fps, w, h)
            yield {"stage": "debug", "message": f"Recording via ffmpeg to {saved_basename}"}
        except Exception as e:
            logging.warning('ffmpeg recording failed to start: %s', e)
            ffmpeg_proc = None
            writer = None
            writer_type = None
    
    # Fallback to OpenCV VideoWriter
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(saved_path, fourcc, cap_fps, (w, h))
        writer_type = 'cv2'
        
        # Check if mp4v worked; if not, try AVI/XVID
        if not getattr(writer, 'isOpened', lambda: False)():
            try:
                writer.release()
            except Exception as _e_rel:
                logging.debug('VideoWriter release failed before fallback: %s', _e_rel)
            
            saved_basename = f"live_{uniq}.avi"
            saved_path = os.path.join(processed_dir, saved_basename)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(saved_path, fourcc, cap_fps, (w, h))
            yield {"stage": "debug", "message": "OpenCV mp4v failed; using AVI/XVID fallback"}
        
        if not getattr(writer, 'isOpened', lambda: False)():
            raise RuntimeError('VideoWriter failed to open for mp4 and avi fallbacks')
        
        print('Recording live stream to %s (fps=%s size=%dx%d)', saved_path, cap_fps, w, h)
        yield {"stage": "debug", "message": f"Recording via OpenCV VideoWriter to {saved_basename}"}
    
    return {
        'writer': writer,
        'writer_type': writer_type,
        'ffmpeg_proc': ffmpeg_proc,
        'saved_basename': saved_basename,
        'saved_path': saved_path,
    }

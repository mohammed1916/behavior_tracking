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

# Create module-level logger
logger = logging.getLogger(__name__)


def parse_llm_segments(llm_output: str, all_captions: List[Dict], classifier_mode: str, prompt: Optional[str] = None) -> List[Dict]:
    """Parse LLM timeline segmentation output into structured segments.
    
    Expected format from LLM:
    0.50-2.30: work
    3.10-3.10: using_phone
    3.80-5.20: work
    
    Returns list of segments with start_time, end_time, label, captions, prompt.
    """
    import logging
    
    # Strip out thinking/reasoning blocks and preamble before parsing
    llm_output = llm_output.strip()
    
    # # Extract valid timestamp set from captions FIRST
    # valid_timestamps = set()
    # for c in all_captions or []:
    #     try:
    #         valid_timestamps.add(round(float(c.get('t', 0)), 2))
    #     except:
    #         pass
    
    # logger.debug(f"[PARSE_LLM] Valid timestamps from input: {sorted(valid_timestamps)}")
    logger.debug(f"[PARSE_LLM] Raw LLM output (first 300 chars): {llm_output[:300]}")
    
    # Remove thinking blocks and keep only actual segments
    lines = llm_output.split('\n')
    clean_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Check if this looks like a segment line: X.XX-Y.YY: label
        if re.match(r'^\d+\.?\d*\s*-\s*\d+\.?\d*\s*:\s*\w+', line_stripped):
            clean_lines.append(line_stripped)
    
    llm_output = '\n'.join(clean_lines)
    logger.debug(f"[PARSE_LLM] Cleaned segment lines: {llm_output}")
    
    # Deduplicate incoming captions FIRST by (rounded time, normalized text), preserving order
    def _dedupe_captions(caps: List[Dict]) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        for c in caps or []:
            t_raw = c.get('t')
            if t_raw is None:
                continue
            try:
                t = float(t_raw)
            except Exception:
                continue
            txt = (c.get('caption') or '').strip()
            if not txt:
                continue
            key = (round(t, 2), txt.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({'t': t, 'caption': c.get('caption')})
        return out

    # Dedupe all_captions immediately before any processing
    all_captions = _dedupe_captions(all_captions)
    
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
        logger.warning(f"[PARSE_LLM] No valid segments found in LLM output. Creating fallback segment with all {len(all_captions)} captions.")
        logger.warning(f"[PARSE_LLM] LLM output was: {llm_output[:200]}")
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
                'timeline': '\n'.join([f"<t={c['t']:.2f}> {c['caption']}" for c in all_captions]),
                'prompt': prompt
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
        
        # Skip segments where timestamps don't exist in input (unless only 1 caption or fuzzy match)
        start_rounded = round(start_time, 2)
        end_rounded = round(end_time, 2)
        
        # # Strict matching: require segment boundaries to exactly match known timestamps
        # if len(all_captions) > 1:
        #     if start_rounded not in valid_timestamps:
        #         logger.warning(f'[PARSE_LLM] Skipping segment with invalid start time: {start_time} (not in {sorted(valid_timestamps)})')
        #         continue
        #     if end_rounded not in valid_timestamps:
        #         logger.warning(f'[PARSE_LLM] Skipping segment with invalid end time: {end_time} (not in {sorted(valid_timestamps)})')
        #         continue
        
        # Skip invalid ranges: start must be less than end (reject start > end)
        # if start_time > end_time:
        #     logger.warning(f'[PARSE_LLM] Skipping invalid segment: {start_time}-{end_time} (start > end)')
        #     continue
        
        label = rules_mod.normalize_label_text(label_str, output_mode=classifier_mode)
        
        # Find captions within this time range (with small tolerance)
        tolerance = 0.05  # 50ms tolerance for floating point comparison
        segment_captions = [
            c for c in all_captions 
            # if start_time <= c['t'] <= end_time
        ]
        # No need to dedupe again - all_captions is already deduped at entry
        
        if segment_captions:
            logger.info(f'[PARSE_LLM] Created segment {start_time:.2f}-{end_time:.2f}: {label} with {len(segment_captions)} captions')
            segments.append({
                'stage': 'segment',
                'start_time': start_time,
                'end_time': end_time,
                'label': label,
                'captions': [c['caption'] for c in segment_captions],
                'timestamps': [c['t'] for c in segment_captions],
                'duration': end_time - start_time,
                'timeline': '\n'.join([f"<t={c['t']:.2f}> {c['caption']}" for c in segment_captions]),
                'prompt': prompt
            })
    
    # Remove duplicate/overlapping segments (LLM sometimes repeats ranges or fuzzy snapping
    # converges multiple lines to identical boundaries). De-dup by (start,end,label).
    deduplicated: List[Dict] = []
    seen_ranges = set()
    dup_count = 0

    # for seg in segments:
    #     start = float(seg['start_time'])
    #     end = float(seg['end_time'])
    #     lbl = (seg.get('label') or '').lower()
    #     # Key with rounded boundaries and label to prevent cross-label collisions
    #     time_key = (round(start, 2), round(end, 2), lbl)
    #     if time_key not in seen_ranges:
    #         deduplicated.append(seg)
    #         seen_ranges.add(time_key)
    #     else:
    #         dup_count += 1

    # if dup_count > 0:
    #     logger.debug(f"[PARSE_LLM] Removed {dup_count} duplicate segment(s) in window")

    # return deduplicated
    return segments


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

    # Helper to dedupe caption/timestamp pairs with aggressive boundary matching
    def _dedupe_pairs(ts: List[Any], caps: List[str]) -> Tuple[List[Any], List[str]]:
        """Remove duplicate (timestamp, caption) pairs.
        
        Two entries are considered duplicates if:
        - Timestamps are within 0.1s of each other AND
        - Caption text matches (case-insensitive, trimmed)
        """
        out_ts: List[Any] = []
        out_caps: List[str] = []
        
        for i, (t, c) in enumerate(zip(ts or [], caps or [])):
            try:
                curr_t = float(t)
            except Exception:
                continue
            curr_c = (c or '').strip()
            if not curr_c:
                continue
            curr_c_norm = curr_c.lower()
            
            # Check if this (t, c) is a duplicate of any already-added entry
            is_dup = False
            for prev_t, prev_c in zip(out_ts, out_caps):
                try:
                    prev_t_f = float(prev_t)
                except Exception:
                    continue
                prev_c_norm = (prev_c or '').strip().lower()
                
                # Consider duplicate if timestamps within 0.1s and text matches
                if abs(curr_t - prev_t_f) <= 0.1 and curr_c_norm == prev_c_norm:
                    is_dup = True
                    break
            
            if not is_dup:
                out_ts.append(t)
                out_caps.append(c)
        
        return out_ts, out_caps

    # Pre-merge pending with first new segment if label matches and gap small
    if pending_segment and segs:
        head = segs[0]
        gap = head['start_time'] - pending_segment['end_time']
        if pending_segment.get('label') == head.get('label') and gap <= merge_gap_sec:
            pending_segment['end_time'] = head['end_time']
            pending_segment['duration'] = pending_segment['end_time'] - pending_segment['start_time']
            caps = (pending_segment.get('captions', []) or []) + (head.get('captions', []) or [])
            ts = (pending_segment.get('timestamps', []) or []) + (head.get('timestamps', []) or [])
            ts, caps = _dedupe_pairs(ts, caps)
            pending_segment['captions'] = caps
            pending_segment['timestamps'] = ts
            pending_segment['timeline'] = '\n'.join([f"<t={float(t):.2f}> {c}" for t, c in zip(ts, caps)])
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


def probe_saved_video_info(saved_path: str, default_fps: float, frame_count: int, duration: float) -> Tuple[Dict[str, Any], Optional[str]]:
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
            logger.warning('ffmpeg recording failed to start: %s', e)
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
                logger.debug('VideoWriter release failed before fallback: %s', _e_rel)
            
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

import os
import sqlite3
import json
from datetime import datetime
import uuid
import logging
from backend.vector_store import STORE as vector_store
from backend.llm import get_local_text_llm, TASK_COMPLETION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'analyses.db')
_db_initialized = False

def calculate_work_idle_stats(samples, duration, fps_hint=None):
    """Calculate work/idle duration and percentages from samples.
    
    Args:
        samples: List of sample dicts with 'time_sec' and 'label' keys
        duration: Total video duration in seconds (may be None/0)
        fps_hint: Optional FPS from video metadata
        
    Returns:
        Dict with work_duration_sec, idle_duration_sec, work_percentage, idle_percentage
    """
    if not samples:
        return {
            'work_duration_sec': 0.0,
            'idle_duration_sec': 0.0,
            'work_percentage': 0.0,
            'idle_percentage': 0.0,
            'unclassified_duration_sec': 0.0,
            'unclassified_percentage': 0.0
        }

    # Fallback duration from samples if missing
    sample_max_time = max((s.get('time_sec') or 0) for s in samples) if samples else 0
    total_duration = duration or sample_max_time or 0
    # Guard against zero to avoid div-by-zero; use 1s minimum
    total_duration = total_duration if total_duration > 0 else 1.0

    # Group frames by label
    work_frames = [s['frame_index'] for s in samples if s.get('label') == 'work']
    idle_frames = [s['frame_index'] for s in samples if s.get('label') == 'idle']

    # Estimate fps: prefer hint, else derive from frame/time relation, else default 30
    fps = fps_hint or None
    if fps is None or fps <= 0:
        # try derive using max frame / duration
        max_frame = max((s.get('frame_index') or 0) for s in samples) if samples else 0
        fps = (max_frame / total_duration) if total_duration > 0 else 0
    if fps is None or fps <= 0:
        fps = 30.0

    work_ranges = compute_ranges(work_frames, samples, fps) if work_frames else []
    idle_ranges = compute_ranges(idle_frames, samples, fps) if idle_frames else []

    work_duration = sum(max(0, r.get('endTime', 0) - r.get('startTime', 0)) for r in work_ranges)
    idle_duration = sum(max(0, r.get('endTime', 0) - r.get('startTime', 0)) for r in idle_ranges)

    work_percentage = (work_duration / total_duration * 100) if total_duration > 0 else 0.0
    idle_percentage = (idle_duration / total_duration * 100) if total_duration > 0 else 0.0
    
    # Calculate unclassified time (gaps between labeled segments)
    classified_duration = work_duration + idle_duration
    unclassified_duration = max(0, total_duration - classified_duration)
    unclassified_percentage = (unclassified_duration / total_duration * 100) if total_duration > 0 else 0.0

    stats = {
        'work_duration_sec': round(work_duration, 2),
        'idle_duration_sec': round(idle_duration, 2),
        'work_percentage': round(work_percentage, 1),
        'idle_percentage': round(idle_percentage, 1),
        'unclassified_duration_sec': round(unclassified_duration, 2),
        'unclassified_percentage': round(unclassified_percentage, 1)
    }

    logger.info(
        "[STATS] samples=%d duration=%.2f fps=%.2f work=%.2fs idle=%.2fs unclass=%.2fs work%%=%.1f idle%%=%.1f unclass%%=%.1f",
        len(samples), total_duration, fps, stats['work_duration_sec'], stats['idle_duration_sec'],
        stats['unclassified_duration_sec'], stats['work_percentage'], stats['idle_percentage'],
        stats['unclassified_percentage']
    )

    return stats


def _stats_from_segments(segments, duration):
    """Fallback stats computation when only segments are available."""
    if not segments:
        return None

    work_dur = sum(max(0.0, float(s.get('duration') or 0.0)) for s in segments if (s.get('label') or '').lower() == 'work')
    idle_dur = sum(max(0.0, float(s.get('duration') or 0.0)) for s in segments if (s.get('label') or '').lower() == 'idle')

    max_end = max((float(s.get('end_time') or 0.0) for s in segments), default=0.0)
    total_duration = duration or max_end or 0.0
    total_duration = total_duration if total_duration > 0 else 1.0

    classified_dur = work_dur + idle_dur
    unclassified_dur = max(0, total_duration - classified_dur)
    
    stats = {
        'work_duration_sec': round(work_dur, 2),
        'idle_duration_sec': round(idle_dur, 2),
        'work_percentage': round((work_dur / total_duration) * 100, 1),
        'idle_percentage': round((idle_dur / total_duration) * 100, 1),
        'unclassified_duration_sec': round(unclassified_dur, 2),
        'unclassified_percentage': round((unclassified_dur / total_duration) * 100, 1),
    }

    logger.info(
        "[STATS_SEGMENTS] segs=%d duration=%.2f work=%.2fs idle=%.2fs unclass=%.2fs work%%=%.1f idle%%=%.1f unclass%%=%.1f",
        len(segments), total_duration, stats['work_duration_sec'], stats['idle_duration_sec'],
        stats['unclassified_duration_sec'], stats['work_percentage'], stats['idle_percentage'],
        stats['unclassified_percentage']
    )
    return stats

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
        created_at TEXT,
        work_duration_sec REAL,
        idle_duration_sec REAL,
        work_percentage REAL,
        idle_percentage REAL,
        unclassified_duration_sec REAL,
        unclassified_percentage REAL
    )
    ''')
    # Backward-compat: add missing stats columns if the DB was created before these fields existed.
    try:
        cur.execute('PRAGMA table_info(analyses)')
        cols = {row[1] for row in cur.fetchall()}
        missing_stats = [
            ('work_duration_sec', 'REAL', 0),
            ('idle_duration_sec', 'REAL', 0),
            ('work_percentage', 'REAL', 0),
            ('idle_percentage', 'REAL', 0),
            ('unclassified_duration_sec', 'REAL', 0),
            ('unclassified_percentage', 'REAL', 0),
        ]
        for name, typ, default_val in missing_stats:
            if name not in cols:
                cur.execute(f"ALTER TABLE analyses ADD COLUMN {name} {typ} DEFAULT {default_val}")
    except Exception:
        # If migration fails, leave DB as-is to avoid breaking existing data
        pass
    # Tasks and subtasks are stored in the vector store (backend/vector_store.py)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT,
        frame_index INTEGER,
        time_sec REAL,
        caption TEXT,
        label TEXT,
        llm_output TEXT,
        detector_metadata TEXT,
        FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    )
    ''')
    # Add detector_metadata column if missing (backward compatibility)
    try:
        cur.execute('PRAGMA table_info(samples)')
        cols = {row[1] for row in cur.fetchall()}
        if 'detector_metadata' not in cols:
            cur.execute("ALTER TABLE samples ADD COLUMN detector_metadata TEXT")
    except Exception:
        # If migration fails, leave DB as-is to avoid breaking existing data
        pass
    cur.execute('''
    CREATE TABLE IF NOT EXISTS segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT,
        start_time REAL,
        end_time REAL,
        duration REAL,
        timeline TEXT,
        label TEXT,
        llm_output TEXT,
        prompt TEXT,
        FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    )
    ''')
    conn.commit()
    conn.close()
    _db_initialized = True

def save_analysis_to_db(analysis_id, filename, model, prompt, video_url, video_info, samples, subtask_id=None, segments=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created = datetime.utcnow().isoformat() + 'Z'
    
    # Calculate work/idle statistics (use duration from metadata or fallback to samples)
    duration = video_info.get('duration') if video_info else 0
    fps_hint = video_info.get('fps') if video_info else None
    # Prefer labeled samples; if missing, fall back to segments-based stats
    labeled_samples = [s for s in (samples or []) if (s.get('label') or '').lower() in ('work', 'idle')]
    if labeled_samples:
        stats = calculate_work_idle_stats(labeled_samples, duration or 0, fps_hint=fps_hint)
    else:
        stats = _stats_from_segments(segments or [], duration or 0) or calculate_work_idle_stats([], duration or 0, fps_hint=fps_hint)
    logger.info(
        "[STATS_SAVE] analysis=%s duration=%.2f fps=%.2f work=%.2fs idle=%.2fs unclass=%.2fs work%%=%.1f idle%%=%.1f unclass%%=%.1f",
        analysis_id, duration or 0, fps_hint or 0,
        stats['work_duration_sec'], stats['idle_duration_sec'], stats['unclassified_duration_sec'],
        stats['work_percentage'], stats['idle_percentage'], stats['unclassified_percentage']
    )
    
    cur.execute('''INSERT OR REPLACE INTO analyses (id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at, work_duration_sec, idle_duration_sec, work_percentage, idle_percentage, unclassified_duration_sec, unclassified_percentage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        analysis_id, filename, model, prompt, video_url,
        video_info.get('fps') if video_info else None,
        video_info.get('frame_count') if video_info else None,
        video_info.get('width') if video_info else None,
        video_info.get('height') if video_info else None,
        duration,
        subtask_id,
        created,
        stats['work_duration_sec'],
        stats['idle_duration_sec'],
        stats['work_percentage'],
        stats['idle_percentage'],
        stats['unclassified_duration_sec'],
        stats['unclassified_percentage']
    ))
    logger.info("[DB_SAVE] analysis %s saved to database", analysis_id)
    if samples:
        logger.info("[DB_SAVE] analysis %s saving %d samples to database", analysis_id, len(samples))
        for s in samples:
            detector_metadata_json = None
            if 'detector_metadata' in s and s['detector_metadata']:
                detector_metadata_json = json.dumps(s['detector_metadata'])
            cur.execute('''INSERT or REPLACE INTO samples (analysis_id, frame_index, time_sec, caption, label, llm_output, detector_metadata) VALUES (?, ?, ?, ?, ?, ?, ?)''', (
                analysis_id,
                s.get('frame_index'),
                s.get('time_sec'),
                s.get('caption'),
                s.get('label'),
                s.get('llm_output'),
                detector_metadata_json
            ))
    logger.info("[DB_SAVE] analysis %s samples saved to database", analysis_id)
    if segments:
        logger.info("[DB_SAVE] analysis %s saving %d segments to database", analysis_id, len(segments))
        for seg in segments:
            cur.execute('''INSERT or REPLACE INTO segments (analysis_id, start_time, end_time, duration, timeline, label, llm_output, prompt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                analysis_id,
                seg.get('start_time'),
                seg.get('end_time'),
                seg.get('duration'),
                seg.get('timeline'),
                seg.get('label'),
                seg.get('llm_output'),
                seg.get('prompt')
            ))
    logger.info("[DB_SAVE_COMPLETE] analysis %s saved to database", analysis_id)
    conn.commit()
    conn.close()

def list_analyses_from_db(limit=100):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at, work_duration_sec, idle_duration_sec, work_percentage, idle_percentage, unclassified_duration_sec, unclassified_percentage FROM analyses ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'id': r[0], 'filename': r[1], 'model': r[2], 'prompt': r[3], 'video_url': r[4],
            'fps': r[5], 'frame_count': r[6], 'width': r[7], 'height': r[8], 'duration': r[9],
            'subtask_id': r[10], 'created_at': r[11],
            'work_duration_sec': r[12], 'idle_duration_sec': r[13],
            'work_percentage': r[14], 'idle_percentage': r[15],
            'unclassified_duration_sec': r[16], 'unclassified_percentage': r[17]
        })
    return out

def get_analysis_from_db(aid):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, filename, model, prompt, video_url, fps, frame_count, width, height, duration, subtask_id, created_at, work_duration_sec, idle_duration_sec, work_percentage, idle_percentage, unclassified_duration_sec, unclassified_percentage FROM analyses WHERE id=?', (aid,))
    r = cur.fetchone()
    if not r:
        conn.close()
        return None
    analysis = {
        'id': r[0], 'filename': r[1], 'model': r[2], 'prompt': r[3], 'video_url': r[4],
        'fps': r[5], 'frame_count': r[6], 'width': r[7], 'height': r[8], 'duration': r[9],
        'subtask_id': r[10], 'created_at': r[11],
        'work_duration_sec': r[12], 'idle_duration_sec': r[13],
        'work_percentage': r[14], 'idle_percentage': r[15],
        'unclassified_duration_sec': r[16], 'unclassified_percentage': r[17]
    }
    cur.execute('SELECT frame_index, time_sec, caption, label, llm_output, detector_metadata FROM samples WHERE analysis_id=? ORDER BY frame_index ASC', (aid,))
    rows = cur.fetchall()
    samples = []
    idle_frames = []
    work_frames = []
    for s in rows:
        detector_metadata = None
        if s[5]:  # detector_metadata column
            try:
                detector_metadata = json.loads(s[5])
            except:
                detector_metadata = None
        sample = {
            'frame_index': s[0], 
            'time_sec': s[1], 
            'caption': s[2], 
            'label': s[3], 
            'llm_output': s[4],
            'detector_metadata': detector_metadata
        }
        samples.append(sample)
        if s[3] == 'idle':
            idle_frames.append(s[0])
        elif s[3] == 'work':
            work_frames.append(s[0])
    analysis['samples'] = samples
    analysis['idle_frames'] = idle_frames
    analysis['work_frames'] = work_frames
    try:
        fps = analysis.get('fps') or 30.0
        analysis['idle_ranges'] = compute_ranges(idle_frames, samples, fps)
        analysis['work_ranges'] = compute_ranges(work_frames, samples, fps)
    except Exception:
        analysis['idle_ranges'] = []
        analysis['work_ranges'] = []
    # Retrieve segments if they exist
    cur.execute('SELECT start_time, end_time, duration, timeline, label, llm_output, prompt FROM segments WHERE analysis_id=? ORDER BY start_time ASC', (aid,))
    seg_rows = cur.fetchall()
    segments = []
    for seg in seg_rows:
        segments.append({
            'start_time': seg[0],
            'end_time': seg[1],
            'duration': seg[2],
            'timeline': seg[3],
            'label': seg[4],
            'llm_output': seg[5],
            'prompt': seg[6]
        })
    analysis['segments'] = segments

    # If sample-level labels were not produced (e.g., classifier_source == 'llm'),
    # synthesize work/idle ranges and representative frame indices from segments so
    # the UI can render seekable segments just like the VLM-only path.
    try:
        if segments:
            fps = analysis.get('fps') or 30.0
            frame_count = analysis.get('frame_count') or None

            def _to_range(seg):
                st = float(seg.get('start_time') or 0.0)
                et = float(seg.get('end_time') or st)
                if et < st:
                    et = st
                start_f = int(round(st * fps))
                end_f = int(round(et * fps))
                if frame_count is not None:
                    start_f = max(0, min(start_f, max(frame_count - 1, 0)))
                    end_f = max(start_f, min(end_f, max(frame_count - 1, 0)))
                return {
                    'startTime': st,
                    'endTime': et,
                    'startFrame': start_f,
                    'endFrame': end_f,
                }

            work_ranges_seg = [_to_range(seg) for seg in segments if (seg.get('label') or '').lower() == 'work']
            idle_ranges_seg = [_to_range(seg) for seg in segments if (seg.get('label') or '').lower() == 'idle']

            if not analysis.get('work_ranges'):
                analysis['work_ranges'] = work_ranges_seg
            if not analysis.get('idle_ranges'):
                analysis['idle_ranges'] = idle_ranges_seg

            step = max(1, int(round(fps)))  # ~1 frame per second to keep arrays small

            def _frames_from_ranges(ranges):
                frames = []
                for r in ranges:
                    start_f = r.get('startFrame', 0)
                    end_f = r.get('endFrame', start_f)
                    seq = list(range(start_f, end_f + 1, step)) if end_f >= start_f else [start_f]
                    if end_f not in seq:
                        seq.append(end_f)
                    frames.extend(seq)
                # keep unique and sorted
                return sorted({int(f) for f in frames})

            if not analysis.get('work_frames'):
                analysis['work_frames'] = _frames_from_ranges(work_ranges_seg)
            if not analysis.get('idle_frames'):
                analysis['idle_frames'] = _frames_from_ranges(idle_ranges_seg)
    except Exception:
        pass

    conn.close()
    return analysis

def delete_analysis_from_db(aid):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('DELETE FROM samples WHERE analysis_id=?', (aid,))
    cur.execute('DELETE FROM segments WHERE analysis_id=?', (aid,))
    cur.execute('DELETE FROM analyses WHERE id=?', (aid,))
    conn.commit()
    conn.close()

def save_task_to_db(task_id, name):
    # Persist a task into the vector store. 'name' is stored as text and also used
    # to derive a deterministic pseudo-embedding.
    created = datetime.utcnow().isoformat() + 'Z'
    metadata = {'name': name, 'created_at': created}
    vector_store.upsert('tasks', task_id, text=name, metadata=metadata)

def list_tasks_from_db():
    # Return all tasks from the vector store sorted by created_at desc
    items = vector_store.list('tasks')
    out = []
    for it in items:
        meta = it.get('metadata', {})
        out.append({'id': it.get('id'), 'name': meta.get('name'), 'created_at': meta.get('created_at')})
    out.sort(key=lambda x: x.get('created_at') or '', reverse=True)
    return out

def get_task_from_db(task_id):
    it = vector_store.get('tasks', task_id)
    if not it:
        return None
    meta = it.get('metadata', {})
    return {'id': it.get('id'), 'name': meta.get('name'), 'created_at': meta.get('created_at')}

def save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time=0, completed_with_delay=0):
    # Save subtask into vector store; include task_id reference in metadata
    created = datetime.utcnow().isoformat() + 'Z'
    metadata = {
        'task_id': task_id,
        'subtask_info': subtask_info,
        'duration_sec': duration_sec,
        'completed_in_time': completed_in_time,
        'completed_with_delay': completed_with_delay,
        'created_at': created,
    }
    text = subtask_info or ''
    vector_store.upsert('subtasks', subtask_id, text=text, metadata=metadata)

def list_subtasks_from_db():
    # List subtasks and include parent task name where available
    items = vector_store.list('subtasks')
    out = []
    for it in items:
        meta = it.get('metadata', {})
        task = vector_store.get('tasks', meta.get('task_id')) if meta.get('task_id') else None
        task_name = (task.get('metadata', {}).get('name')) if task else None
        out.append({
            'id': it.get('id'),
            'task_id': meta.get('task_id'),
            'task_name': task_name,
            'subtask_info': meta.get('subtask_info'),
            'duration_sec': meta.get('duration_sec'),
            'completed_in_time': meta.get('completed_in_time'),
            'completed_with_delay': meta.get('completed_with_delay'),
            'created_at': meta.get('created_at'),
        })
    out.sort(key=lambda x: x.get('created_at') or '', reverse=True)
    return out

def get_subtask_from_db(subtask_id):
    it = vector_store.get('subtasks', subtask_id)
    if not it:
        return None
    meta = it.get('metadata', {})
    task = vector_store.get('tasks', meta.get('task_id')) if meta.get('task_id') else None
    task_name = (task.get('metadata', {}).get('name')) if task else None
    return {
        'id': it.get('id'),
        'task_id': meta.get('task_id'),
        'task_name': task_name,
        'subtask_info': meta.get('subtask_info'),
        'duration_sec': meta.get('duration_sec'),
        'completed_in_time': meta.get('completed_in_time'),
        'completed_with_delay': meta.get('completed_with_delay'),
        'created_at': meta.get('created_at'),
    }


def delete_subtask_from_db(subtask_id):
    return vector_store.delete('subtasks', subtask_id)


def delete_task_from_db(task_id):
    # delete subtasks referencing this task, then delete the task
    subs = vector_store.list('subtasks')
    for it in subs:
        if it.get('metadata', {}).get('task_id') == task_id:
            vector_store.delete('subtasks', it.get('id'))
    return vector_store.delete('tasks', task_id)

def update_samples_label(analysis_id, label, start_frame=None, end_frame=None, start_time=None, end_time=None):
    """Update labels for a range of samples in an analysis."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # helper to find frames by time if needed
    if start_time is not None and start_frame is None:
        # get FPS from analysis
        cur.execute('SELECT fps FROM analyses WHERE id=?', (analysis_id,))
        r = cur.fetchone()
        fps = r[0] if r else 30.0
        start_frame = int(start_time * fps)
        if end_time is not None:
            end_frame = int(end_time * fps)
            
    query = 'UPDATE samples SET label=? WHERE analysis_id=?'
    params = [label, analysis_id]
    
    if start_frame is not None:
        query += ' AND frame_index >= ?'
        params.append(start_frame)
    if end_frame is not None:
        query += ' AND frame_index <= ?'
        params.append(end_frame)
        
    cur.execute(query, tuple(params))
    rows = cur.rowcount
    conn.commit()
    conn.close()
    return rows

def compute_ranges(frames, samples, fps):
    if not frames or (hasattr(frames, '__len__') and len(frames) == 0):
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
    for r in ranges:
        captions = [s['caption'] for s in (samples or []) if 'time_sec' in s and r['startTime'] - 0.0001 <= s['time_sec'] <= r['endTime'] + 0.0001 and s.get('caption')]
        r['captions'] = captions
    return ranges


def _llm_decision_for_task(info, captions, llm=None, subtask_id=None):
    """Centralize LLM init, prompt building, call and response parsing.

    Returns: (decision_bool, llm_output_str)
    Raises RuntimeError on missing LLM or ambiguous output (development-only).
    """
    if llm is None:
        llm = get_local_text_llm()
        if llm is None:
            raise RuntimeError('Local text LLM not available; ensure the environment provides a working LLM (e.g. ollama)')
    prompt_template = TASK_COMPLETION_PROMPT_TEMPLATE
    prompt = prompt_template.format(task=info or '', captions=captions or '')
    resp = llm(prompt)
    if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
        llm_output = resp[0].get('generated_text', '').strip()
    else:
        llm_output = str(resp)
    # Log raw LLM output and captions preview
    # print('LLM output for subtask %s: %s', subtask_id or '<anon>', (llm_output or '').replace('\n', ' '))
    # logger.debug('captions_preview: %s', (captions or '')[:1000])
    txt = (llm_output or '').strip().lower()
    affirmatives = ('y', 'yes', 'true', 'work', 'completed', 'done', 'active', 'activity', 'movement')
    negatives = ('n', 'no', 'false', 'idle', 'not', 'inactive', 'still', 'stationary')
    if any(a in txt for a in affirmatives):
        return True, llm_output
    if any(n in txt for n in negatives):
        return False, llm_output
    # Ambiguous
    raise RuntimeError(f'LLM returned ambiguous decision for subtask {subtask_id or "<anon>"}: "{llm_output}"')


def evaluate_subtasks_completion(captions, llm=None, top_k=5):
    """Evaluate likely subtask completion using the local text LLM and FAISS-backed subtasks.

    captions: list of caption strings (or single string)
    llm: optional callable (prompt -> [{'generated_text': ...}])
    Returns: list of dicts: {subtask_id, task_id, subtask_info, completed (bool), llm_output}
    """
    # normalize captions input
    
    # print("Running evaluate_subtasks_completion with captions:", captions)
    if not captions:
        return []
    if isinstance(captions, (list, tuple)):
        combined = '\n'.join([str(c) for c in captions])
    else:
        combined = str(captions)

    # Require a functioning local LLM during development - fail fast.
    if llm is None:
        llm = get_local_text_llm()
        if llm is None:
            raise RuntimeError('Local text LLM not available; ensure the environment provides a working LLM (e.g. ollama)')
    prompt_template = TASK_COMPLETION_PROMPT_TEMPLATE

    # find candidate subtasks via vector search
    candidates = vector_store.search('subtasks', query_text=combined, top_k=top_k)

    results = []
    for cand in candidates:
        cid = cand.get('id')
        meta = cand.get('metadata', {})
        task_id = meta.get('task_id')
        info = meta.get('subtask_info') or meta.get('name') or ''
        # Ask LLM for decision (centralized helper will init LLM and parse)
        completed, llm_out = _llm_decision_for_task(info, combined, llm=llm, subtask_id=cid)
        results.append({'subtask_id': cid, 'task_id': task_id, 'subtask_info': info, 'completed': completed, 'llm_output': llm_out})

    return results

def update_subtask_counts(subtask_id, completed_in_time_increment, completed_with_delay_increment):
    it = vector_store.get('subtasks', subtask_id)
    if not it:
        return False
    meta = it.get('metadata', {})
    meta['completed_in_time'] = int(meta.get('completed_in_time', 0)) + int(completed_in_time_increment)
    meta['completed_with_delay'] = int(meta.get('completed_with_delay', 0)) + int(completed_with_delay_increment)
    # persist updated metadata
    vector_store.upsert('subtasks', subtask_id, text=it.get('text'), metadata=meta, vector=it.get('vector'))
    return True


def aggregate_and_update_subtask(subtask_id, collected_samples, collected_work, fps, llm=None, policy='default', window_sec=5.0, padding_sec=1.0, min_segment_sec=0.5, merge_gap_sec=1.0):
    """Aggregate LLM + timing signals for a specific subtask and update counts.

    Returns (completed_in_time_increment, completed_with_delay_increment, reason)
    """
    # fetch subtask
    it = vector_store.get('subtasks', subtask_id)
    if not it:
        return (0, 0, 'subtask_not_found')
    meta = it.get('metadata', {})
    expected = meta.get('duration_sec')
    if expected is None:
        return (0, 0, 'no_expected_duration')

    # compute work ranges from provided frames and samples
    try:
        ranges = compute_ranges(collected_work or [], collected_samples or [], fps or 30.0)
    except Exception:
        ranges = []

    # merge and filter ranges
    def _merge_ranges(rs):
        if not rs:
            return []
        rs_sorted = sorted(rs, key=lambda r: r.get('startTime', 0))
        merged = []
        cur = rs_sorted[0].copy()
        for r in rs_sorted[1:]:
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

    merged = _merge_ranges(ranges)
    actual_work_time = sum((r.get('endTime', 0) - r.get('startTime', 0)) for r in merged)

    # prepare LLM
    # Require a functioning local LLM during debug; fail fast if missing.
    if llm is None:
        llm = get_local_text_llm()
        if llm is None:
            raise RuntimeError('Local text LLM not available; ensure the environment provides a working LLM (e.g. ollama)')
    prompt_template = TASK_COMPLETION_PROMPT_TEMPLATE

    # collect captions within windows around each merged range (or fallback to all captions)
    captions_for_llm = ''
    if merged:
        pieces = []
        for r in merged:
            start_w = max(0.0, r.get('startTime', 0) - padding_sec)
            end_w = r.get('endTime', 0) + padding_sec
            # gather captions within window
            caps = [s.get('caption') for s in (collected_samples or []) if s.get('time_sec') is not None and start_w <= s.get('time_sec') <= end_w and s.get('caption')]
            if caps:
                pieces.append('\n'.join(caps))
        captions_for_llm = '\n\n'.join(pieces)
    else:
        # no merged ranges: fallback to all captions
        captions_for_llm = '\n'.join([s.get('caption') for s in (collected_samples or []) if s.get('caption')])

    llm_decision = None
    llm_output = None
    # Require LLM/prompt template; do not fall back to heuristics in development.
    if not captions_for_llm:
        raise RuntimeError('No captions available for LLM evaluation')

    # Use centralized helper to get LLM decision and output
    llm_decision, llm_output = _llm_decision_for_task(meta.get('subtask_info') or '', captions_for_llm, llm=llm, subtask_id=subtask_id)

    # aggregation policy (default)
    completed_in = 0
    completed_delay = 0
    reason = 'none'
    if llm_decision is True:
        if actual_work_time <= float(expected):
            completed_in = 1
            reason = 'llm_yes_in_time'
        else:
            completed_delay = 1
            reason = 'llm_yes_with_delay'
    elif llm_decision is False:
        # LLM says not completed -> fall back to timing if within expected
        if actual_work_time <= float(expected):
            completed_in = 1
            reason = 'llm_no_but_timing_in_time'
        else:
            reason = 'llm_no_timing_exceeds'
    else:
        # LLM uncertain -> fallback to timing
        if actual_work_time <= float(expected):
            completed_in = 1
            reason = 'llm_uncertain_timing_in_time'
        else:
            reason = 'llm_uncertain_timing_exceeds'

    if completed_in or completed_delay:
        update_subtask_counts(subtask_id, completed_in, completed_delay)

    return (completed_in, completed_delay, reason)
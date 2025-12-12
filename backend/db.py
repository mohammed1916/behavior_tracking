import os
import sqlite3
import json
from datetime import datetime
import uuid
import logging

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
        pass
    try:
        cur.execute('ALTER TABLE subtasks ADD COLUMN completed_with_delay INTEGER DEFAULT 0;')
    except sqlite3.OperationalError:
        pass
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
    try:
        fps = analysis.get('fps') or 30.0
        analysis['idle_ranges'] = compute_ranges(idle_frames, samples, fps)
        analysis['work_ranges'] = compute_ranges(work_frames, samples, fps)
    except Exception:
        analysis['idle_ranges'] = []
        analysis['work_ranges'] = []
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
    cur.execute('''INSERT OR REPLACE INTO tasks (id, name, created_at) VALUES (?, ?, ?)''', (task_id, name, created))
    conn.commit()
    conn.close()

def list_tasks_from_db():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, created_at FROM tasks ORDER BY created_at DESC')
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'name': r[1], 'created_at': r[2]} for r in rows]

def get_task_from_db(task_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('SELECT id, name, created_at FROM tasks WHERE id=?', (task_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {'id': r[0], 'name': r[1], 'created_at': r[2]}

def save_subtask_to_db(subtask_id, task_id, subtask_info, duration_sec, completed_in_time=0, completed_with_delay=0):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created = datetime.utcnow().isoformat() + 'Z'
    cur.execute('''INSERT OR REPLACE INTO subtasks (id, task_id, subtask_info, duration_sec, completed_in_time, completed_with_delay, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)''', (subtask_id, task_id, subtask_info, duration_sec, completed_in_time, completed_with_delay, created))
    conn.commit()
    conn.close()

def list_subtasks_from_db():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''SELECT s.id, t.name, s.subtask_info, s.duration_sec, s.completed_in_time, s.completed_with_delay, s.created_at FROM subtasks s JOIN tasks t ON s.task_id = t.id ORDER BY s.created_at DESC''')
    rows = cur.fetchall()
    conn.close()
    return [{'id': r[0], 'task_name': r[1], 'subtask_info': r[2], 'duration_sec': r[3], 'completed_in_time': r[4], 'completed_with_delay': r[5], 'created_at': r[6]} for r in rows]

def get_subtask_from_db(subtask_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''SELECT s.id, t.name, s.subtask_info, s.duration_sec, s.completed_in_time, s.completed_with_delay, s.created_at FROM subtasks s JOIN tasks t ON s.task_id = t.id WHERE s.id = ?''', (subtask_id,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {'id': r[0], 'task_name': r[1], 'subtask_info': r[2], 'duration_sec': r[3], 'completed_in_time': r[4], 'completed_with_delay': r[5], 'created_at': r[6]}

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

def update_subtask_counts(subtask_id, completed_in_time_increment, completed_with_delay_increment):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('UPDATE subtasks SET completed_in_time = completed_in_time + ?, completed_with_delay = completed_with_delay + ? WHERE id = ?', (completed_in_time_increment, completed_with_delay_increment, subtask_id))
    conn.commit()
    conn.close()

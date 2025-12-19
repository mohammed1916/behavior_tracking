import os
import sys
import uuid
import pytest

# Ensure repo root is on sys.path so `backend` package imports resolve during pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend import db as db_mod
import types


def ensure_llm_prompt_template():
    # Ensure a minimal `llm` module with TASK_COMPLETION_PROMPT_TEMPLATE is available
    if 'llm' in sys.modules:
        return
    m = types.SimpleNamespace()
    m.TASK_COMPLETION_PROMPT_TEMPLATE = "Task: {task}\nCaptions:\n{captions}\nDid the task complete?"
    def get_local_text_llm():
        def _dummy(prompt):
            return [{'generated_text': 'uncertain'}]
        return _dummy
    m.get_local_text_llm = get_local_text_llm
    sys.modules['llm'] = m


# Ensure llm prompt template is present so aggregate function uses provided llm mocks
ensure_llm_prompt_template()


def make_samples_for_range(start_sec, end_sec, count=3):
    """Produce evenly spaced sample dicts for given time window."""
    if count < 1:
        return []
    dur = max(0.0001, end_sec - start_sec)
    step = dur / float(count)
    samples = []
    for i in range(count):
        t = start_sec + i * step
        # use arbitrary frame indices increasing
        samples.append({'frame_index': int(t * 30), 'time_sec': t, 'caption': f'cap_{t:.2f}', 'label': 'work'})
    return samples


def test_aggregation_llm_yes_in_time():
    tid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    # subtask duration expected 5s
    db_mod.save_task_to_db(tid, 'test task')
    db_mod.save_subtask_to_db(sid, tid, 'do thing', 5)

    # create samples representing a 3s contiguous work range
    samples = make_samples_for_range(10.0, 13.0, count=4)
    work_frames = [s['frame_index'] for s in samples]

    # mock llm that returns 'yes'
    def llm_yes(prompt):
        return [{'generated_text': 'yes'}]

    inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(sid, samples, work_frames, fps=30.0, llm=llm_yes)
    assert inc_in == 1 and inc_delay == 0
    assert reason.startswith('llm_yes')

    # confirm persisted metadata incremented
    s = db_mod.get_subtask_from_db(sid)
    assert s['completed_in_time'] >= 1


def test_aggregation_llm_yes_with_delay():
    tid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    db_mod.save_task_to_db(tid, 'task2')
    db_mod.save_subtask_to_db(sid, tid, 'do big thing', 2)

    # create samples representing a 5s contiguous work range (exceeds expected 2s)
    samples = make_samples_for_range(20.0, 25.0, count=5)
    work_frames = [s['frame_index'] for s in samples]

    def llm_yes(prompt):
        return [{'generated_text': 'yes'}]

    inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(sid, samples, work_frames, fps=30.0, llm=llm_yes)
    assert inc_in == 0 and inc_delay == 1
    assert reason == 'llm_yes_with_delay'


def test_aggregation_llm_uncertain_fallback_timing():
    tid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    db_mod.save_task_to_db(tid, 'task3')
    db_mod.save_subtask_to_db(sid, tid, 'do medium', 6)

    # create samples representing a 4s contiguous range (within expected)
    samples = make_samples_for_range(30.0, 34.0, count=4)
    work_frames = [s['frame_index'] for s in samples]

    def llm_uncertain(prompt):
        return [{'generated_text': 'maybe'}]

    inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(sid, samples, work_frames, fps=30.0, llm=llm_uncertain)
    assert inc_in == 1 and inc_delay == 0
    assert reason.startswith('llm_uncertain')


def test_aggregation_llm_no_and_timing_exceeds():
    tid = str(uuid.uuid4())
    sid = str(uuid.uuid4())
    db_mod.save_task_to_db(tid, 'task4')
    db_mod.save_subtask_to_db(sid, tid, 'do tiny', 1)

    # create samples representing a 3s contiguous range (exceeds expected)
    samples = make_samples_for_range(40.0, 43.0, count=4)
    work_frames = [s['frame_index'] for s in samples]

    def llm_no(prompt):
        return [{'generated_text': 'no'}]

    inc_in, inc_delay, reason = db_mod.aggregate_and_update_subtask(sid, samples, work_frames, fps=30.0, llm=llm_no)
    assert inc_in == 0 and inc_delay == 0
    assert reason == 'llm_no_timing_exceeds' or reason == 'llm_no_timing_exceeds'

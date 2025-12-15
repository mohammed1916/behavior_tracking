"""
backend.tests.test_timing
pytest -q backend/tests/test_timing.py 
"""
import pytest
import sys, os
# Ensure repo root is on path so `import backend` works when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.db import compute_ranges


def make_samples(times):
    # times: list of (frame_index, time_sec, caption, label)
    return [
        {'frame_index': fi, 'time_sec': ts, 'caption': '', 'label': 'work' if lab else 'idle'}
        for fi, ts, lab in times
    ]


def test_compute_ranges_simple():
    # two work frames close together => one short range
    samples = make_samples([(0, 0.0, True), (1, 0.5, True), (2, 1.5, False)])
    frames = [0,1]
    ranges = compute_ranges(frames, samples, fps=30.0)
    assert len(ranges) == 1
    dur = ranges[0]['endTime'] - ranges[0]['startTime']
    assert dur >= 0.5


def test_compute_ranges_multiple_segments():
    # two separate work segments separated by gap
    samples = make_samples([
        (0, 0.0, True), (1, 0.2, True),
        (10, 5.0, True), (11, 5.4, True)
    ])
    frames = [0,1,10,11]
    ranges = compute_ranges(frames, samples, fps=30.0)
    assert len(ranges) >= 2
    total = sum(r['endTime'] - r['startTime'] for r in ranges)
    assert total > 0

"""
pytest backend/tests/aggregation/test_evidence_window.py -v -s --tb=short
"""
import pytest

from backend.evidence import EvidenceWindow, aggregate_evidence


def make_samples(times, captions=None):
    captions = captions or [f"cap_{i}" for i in range(len(times))]
    return [{"time_sec": t, "caption": c} for t, c in zip(times, captions)]


class TestEvidenceWindowAggregation:
    def test_basic_window(self):
        times = [0.5, 1.0, 1.6]
        samples = make_samples(times, captions=["a", "b", "c"])
        windows = aggregate_evidence(samples, max_window_sec=4.0, silence_timeout_sec=1.5, min_samples=2)
        assert len(windows) == 1
        w = windows[0]
        assert pytest.approx(w.start_time, rel=1e-3) == 0.5
        assert pytest.approx(w.end_time, rel=1e-3) == 1.6
        # Captions never dropped
        assert [s["caption"] for s in w.samples] == ["a", "b", "c"]
        # Timeline format
        tl = w.to_timeline()
        assert tl.count("<t=") == 3
        assert "<t=0.50> a" in tl
        assert "<t=1.60> c" in tl

    def test_split_on_silence(self):
        times = [0.5, 1.0, 1.2, 3.0, 3.3]
        caps = ["cap1", "cap2", "cap3", "cap4", "cap5"]
        windows = aggregate_evidence(make_samples(times, caps), max_window_sec=4.0, silence_timeout_sec=1.5, min_samples=2)
        assert len(windows) == 2
        w1, w2 = windows
        assert [s["caption"] for s in w1.samples] == ["cap1", "cap2", "cap3"]
        assert [s["caption"] for s in w2.samples] == ["cap4", "cap5"]
        # Bounds
        assert w1.start_time == 0.5 and w1.end_time == 1.2
        assert w2.start_time == 3.0 and w2.end_time == 3.3

    def test_close_on_max_window(self):
        # Continuous samples exceeding 3s should close a window
        times = [0.0, 1.0, 2.0, 3.1, 3.8]
        windows = aggregate_evidence(make_samples(times), max_window_sec=3.0, silence_timeout_sec=1.5, min_samples=2)
        # Expect two windows: [0.0, 3.1] and [3.8]
        # Second window has < 2 samples -> discarded
        assert len(windows) == 1
        w = windows[0]
        assert w.start_time == 0.0
        assert w.end_time == 3.1
        assert len(w.samples) == 4

    def test_min_samples_filter(self):
        times = [0.0, 5.0]  # two windows with single sample each if closed by silence
        windows = aggregate_evidence(make_samples(times), max_window_sec=4.0, silence_timeout_sec=1.5, min_samples=2)
        # First window closes immediately when gap > silence; single sample -> discarded
        # Second window closes at end; single sample -> discarded
        assert len(windows) == 0

    def test_ignores_malformed_samples(self):
        samples = [
            {"time_sec": 0.0, "caption": "a"},
            {"time_sec": None, "caption": "bad"},
            {"caption": "missing_time"},
            {"time_sec": 1.0, "caption": None},
            {"time_sec": 1.2, "caption": "b"},
        ]
        windows = aggregate_evidence(samples, min_samples=2)
        assert len(windows) == 1
        w = windows[0]
        assert [s["caption"] for s in w.samples] == ["a", "b"]
        assert w.start_time == 0.0 and w.end_time == 1.2

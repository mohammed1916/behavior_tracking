from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional


@dataclass
class EvidenceWindow:
    """Purely time-based aggregation of VLM samples.

    No semantic similarity. Captions are never dropped.
    """
    start_time: float
    end_time: float
    samples: List[Dict] = field(default_factory=list)

    def add_sample(self, t: float, caption: str):
        self.samples.append({"t": float(t), "caption": caption})
        self.end_time = float(t)

    def to_timeline(self) -> str:
        lines = []
        # Ensure samples are sorted by time (should already be, but being explicit)
        sorted_samples = sorted(self.samples, key=lambda s: float(s['t']))
        for s in sorted_samples:
            lines.append(f"<t={s['t']:.2f}> {s['caption']}")
        return "\n".join(lines)


def aggregate_evidence(samples: Iterable[Dict], *,
                       max_window_sec: float = 4.0,
                       silence_timeout_sec: float = 1.5,
                       min_samples: int = 2) -> List[EvidenceWindow]:
    """Aggregate VLM samples into time-only EvidenceWindows.

    - samples: iterable of {time_sec: float, caption: str}
    - windows close when:
      * gap between consecutive samples > silence_timeout_sec, OR
      * window duration exceeds max_window_sec
    - windows with < min_samples are discarded
    - Captions within windows are preserved with timestamps
    """
    # Normalize and sort by time
    norm = []
    for s in samples:
        t = s.get("time_sec")
        c = s.get("caption")
        if t is None or c is None:
            # ignore malformed sample, do not crash
            continue
        norm.append({"time_sec": float(t), "caption": str(c)})
    norm.sort(key=lambda x: x["time_sec"])  # stable

    out: List[EvidenceWindow] = []
    current: Optional[EvidenceWindow] = None
    prev_t: Optional[float] = None

    def _close_current():
        nonlocal current
        if current is not None:
            if len(current.samples) >= min_samples:
                out.append(current)
            current = None

    for s in norm:
        t = s["time_sec"]
        caption = s["caption"]
        if current is None:
            current = EvidenceWindow(start_time=t, end_time=t, samples=[])
            current.add_sample(t, caption)
            prev_t = t
            continue

        # Determine if we should close due to silence BEFORE adding the sample
        gap = (t - (prev_t if prev_t is not None else t))
        if gap > silence_timeout_sec:
            _close_current()
            current = EvidenceWindow(start_time=t, end_time=t, samples=[])
            current.add_sample(t, caption)
            prev_t = t
            continue

        # Add the sample into the current window
        current.add_sample(t, caption)
        prev_t = t

        # If max_window_sec cap reached or exceeded, close AFTER including this sample
        duration = (current.end_time - current.start_time)
        if duration >= max_window_sec:
            _close_current()

    # close trailing window
    _close_current()

    return out

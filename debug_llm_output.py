#!/usr/bin/env python3
"""Debug script to inspect raw LLM output for VLM+LLM aggregation."""

import sys
import os
import json
import logging
from pathlib import Path

# Setup logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_llm.log')
    ]
)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend import llm as llm_mod
from backend.evidence import EvidenceWindow
from backend.stream_utils import parse_llm_segments

text_llm = llm_mod.get_local_text_llm()
if text_llm is None:
    print("ERROR: No LLM available (Ollama not running)")
    sys.exit(1)

# Create a sample window with 4 captions  
window = EvidenceWindow(start_time=0.0, end_time=4.0)
window.add_sample(0.5, "person holding small object with hands")
window.add_sample(1.2, "hands working on assembly")
window.add_sample(2.3, "person inserting part into component")
window.add_sample(3.8, "hands adjusting electronic component")

timeline = window.to_timeline()
print("=" * 80)
print("TIMELINE INPUT TO LLM:")
print("=" * 80)
print(timeline)
print()

# Format the prompt
rendered = llm_mod.LLM_SEGMENT_TIMELINE_BINARY.format(caption=timeline)
print("=" * 80)
print("FULL PROMPT:")
print("=" * 80)
print(rendered)
print()

# Call LLM
print("=" * 80)
print("CALLING LLM...")
print("=" * 80)
llm_res = text_llm(rendered, max_new_tokens=100)

# Extract output
if isinstance(llm_res, list) and llm_res and isinstance(llm_res[0], dict):
    llm_output = llm_res[0].get('generated_text', str(llm_res))
else:
    llm_output = str(llm_res)

print("\n" + "=" * 80)
print("RAW LLM OUTPUT:")
print("=" * 80)
print(repr(llm_output))
print()
print(llm_output)
print()

# Parse and show results
print("=" * 80)
print("PARSING RESULTS:")
print("=" * 80)
all_captions = [{'t': s['t'], 'caption': s['caption']} for s in window.samples]
segments = parse_llm_segments(llm_output, all_captions, 'llm', prompt=rendered)

print(f"Found {len(segments)} segments:")
for i, seg in enumerate(segments, 1):
    print(f"\nSegment {i}:")
    print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
    print(f"  Label: {seg['label']}")
    print(f"  Captions: {seg['captions']}")
    print(f"  Duration: {seg['duration']:.2f}s")

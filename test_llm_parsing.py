#!/usr/bin/env python3
"""Test script to verify LLM output parsing and deduplication."""

import sys
sys.path.insert(0, 'backend')

from backend.stream_utils import parse_llm_segments
from backend.rules import normalize_label_text

# Test cases for different LLM output scenarios
test_cases = [
    {
        "name": "Clean output",
        "llm_output": """0.50-2.30: work
3.10-3.80: idle
3.80-5.20: work""",
        "expected_count": 3,
        "expected_labels": ["work", "idle", "work"]
    },
    {
        "name": "Output with preamble",
        "llm_output": """Thinking... Let me analyze this timeline.

0.50-2.30: work
3.10-3.80: idle
3.80-5.20: work

That's my analysis.""",
        "expected_count": 3,
        "expected_labels": ["work", "idle", "work"]
    },
    {
        "name": "Duplicate timestamps",
        "llm_output": """0.50-2.30: work
3.80-5.20: work
3.80-5.20: idle""",  # Same time range, different labels - invalid!
        "expected_count": 2,
        "expected_labels": ["work", "work"]  # Should skip duplicate
    },
    {
        "name": "Invalid range (start > end)",
        "llm_output": """0.50-2.30: work
5.20-3.80: idle
6.00-8.50: work""",
        "expected_count": 2,
        "expected_labels": ["work", "work"]  # Should skip invalid
    },
    {
        "name": "Unreasonable times",
        "llm_output": """0.50-2.30: work
43.80-5.20: idle
6.00-8.50: work""",
        "expected_count": 2,
        "expected_labels": ["work", "work"]  # Should skip 43.80 (start > end)
    },
]

# Mock caption data
mock_captions = [
    {'t': 0.50, 'caption': 'Person working on electronics'},
    {'t': 1.30, 'caption': 'Soldering iron in hand'},
    {'t': 2.30, 'caption': 'Continues assembly work'},
    {'t': 3.10, 'caption': 'Puts down tools'},
    {'t': 3.80, 'caption': 'Standing idle'},
    {'t': 5.20, 'caption': 'Back to work'},
    {'t': 6.00, 'caption': 'Working again'},
    {'t': 8.50, 'caption': 'Still working'},
]

print("Testing LLM Output Parsing\n" + "="*60)

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"LLM Output:\n{test['llm_output']}\n")
    
    segments = parse_llm_segments(test['llm_output'], mock_captions, 'binary')
    
    print(f"Parsed {len(segments)} segments:")
    for seg in segments:
        print(f"  {seg['start_time']:.2f}-{seg['end_time']:.2f}: {seg['label']}")
    
    # Check expectations
    if len(segments) == test['expected_count']:
        print(f"✓ Segment count matches: {len(segments)}")
    else:
        print(f"✗ FAIL: Expected {test['expected_count']} segments, got {len(segments)}")
    
    labels = [seg['label'] for seg in segments]
    if labels == test['expected_labels']:
        print(f"✓ Labels match: {labels}")
    else:
        print(f"✗ FAIL: Expected {test['expected_labels']}, got {labels}")

print("\n" + "="*60)
print("Parsing tests complete!")

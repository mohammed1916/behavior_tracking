#!/usr/bin/env python3
"""Test the improved LLM prompt"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend import llm as llm_mod

# Create a sample timeline
timeline = """<t=8.57> there is a man standing in front of a counter with bottles of wine
<t=10.70> there is a man that is standing in front of a machine
<t=12.83> arafed man standing in a factory with a camera in his hand
<t=15.00> there is a man that is standing in a room with a table"""

# Get LLM
text_llm = llm_mod.get_local_text_llm()
if text_llm is None:
    print("ERROR: No LLM available (Ollama not running)")
    sys.exit(1)

# Format prompt
rendered = llm_mod.LLM_SEGMENT_TIMELINE_BINARY.format(caption=timeline)
print("=" * 80)
print("PROMPT SENT TO LLM:")
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

print("\nRAW LLM OUTPUT:")
print(repr(llm_output))
print()
print(llm_output)
print()

# Check if output looks correct
lines = llm_output.strip().split('\n')
segment_lines = [l.strip() for l in lines if '-' in l and ':' in l and any(c.isdigit() for c in l)]
print("=" * 80)
print(f"Extracted {len(segment_lines)} potential segment lines:")
print("=" * 80)
for line in segment_lines:
    print(f"  {line}")

#!/usr/bin/env python
"""
Test to verify that logging with module names works correctly
when stream_utils functions are called.
"""

import logging
import sys

# Set up logging similar to server.py
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s - %(levelname)s %(message)s')

# Test importing the modules
print("Testing logging configuration...")

# Import modules that have loggers
import backend.stream_utils as stream_utils_mod

# Create a test logger from this module
module_logger = logging.getLogger(__name__)
module_logger.info("Test logger from test_logging_output module")

# Test that stream_utils logger works
# Call parse_llm_segments with some dummy data
print("\nTesting parse_llm_segments logging...")

llm_output = """0.50-1.20: work
1.20-2.30: idle
2.30-3.50: work"""

captions = [
    {'t': 0.50, 'caption': 'Person assembling component'},
    {'t': 1.20, 'caption': 'Pauses and checks manual'},
    {'t': 2.30, 'caption': 'Continues assembly'},
    {'t': 3.50, 'caption': 'Finishes task'}
]

result = stream_utils_mod.parse_llm_segments(llm_output, captions, 'work_idle')

print(f"\nParsed {len(result)} segments")
for seg in result:
    print(f"  - {seg['start_time']:.2f}-{seg['end_time']:.2f}: {seg['label']}")

print("\n[SUCCESS] If you see logs from backend.stream_utils above, the logging configuration is working correctly!")
print("  Look for lines like: 'backend.stream_utils - DEBUG [PARSE_LLM]...'")

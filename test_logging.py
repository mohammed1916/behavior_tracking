import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
from backend.stream_utils import parse_llm_segments

# Test with sample data
llm_output = """8.57-10.70: idle
10.70-12.83: work
12.83-15.00: work"""

captions = [
    {'t': 8.57, 'caption': 'man standing by counter'},
    {'t': 10.70, 'caption': 'man at machine'},
    {'t': 12.83, 'caption': 'man in factory'},
    {'t': 15.00, 'caption': 'man at table'}
]

segments = parse_llm_segments(llm_output, captions, 'binary', prompt='test')
print(f'\nParsed {len(segments)} segments')
for seg in segments:
    print(f'  {seg["start_time"]:.2f}-{seg["end_time"]:.2f}: {seg["label"]} ({len(seg["captions"])} captions)')

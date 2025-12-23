import shutil
import subprocess
import logging

# Prompt template for LLM classification of captions (work vs idle).
CLASSIFY_PROMPT_TEMPLATE = (
    "You are a strict classifier. Respond with exactly one lowercase word: either work or idle.\n"
    "Do NOT add any punctuation, explanation, quotes, or extra text — only the single word.\n\n"
    "Context: {prompt}\nCaption: {caption}\n"
)

# Prompt template used by VLM adapters (Qwen / BLIP) for image->label classification.
# Centralising this here allows multiple adapters/scripts to reuse the same instructions.
VLM_BASE_PROMPT_TEMPLATE = """<|vision_start|><|image_pad|><|vision_end|>

You are an expert activity recognition model.

Look ONLY at the MAIN PERSON in the image. Ignore all other people or objects.
"""
LABLE_PROMPT_TEMPLATE_MULTI = """Classify their CURRENT ACTION into exactly ONE label from the following:

1. assembling_drone → The person is working with tools, touching a drone, handling drone parts, connecting wires, tightening screws, or performing assembly actions.
2. idle → The person is standing or sitting without doing any task, arms resting, not interacting with objects.
3. using_phone → The person is clearly holding or interacting with a phone.
4. unknown → If the activity cannot be confidently identified.
- Only output exactly one label: assembling_drone, idle, using_phone, or unknown.
- Do not add any extra text, explanations, or repeats.
"""
LABLE_PROMPT_TEMPLATE_BINARY = """Classify their CURRENT ACTION into exactly ONE label from the following:
1. work → The person is actively engaged in hands-on electronics or drone assembly tasks.
2. idle → The person is not engaged in any task, standing or sitting without interaction.
- Only output exactly one label: work or idle.
- Do not add any extra text, explanations, or repeats.
"""

RULES_PROMPT_TEMPLATE = """Rules:
- Do NOT guess.
- End your answer with "<|endoftext|>"

Answer:
"""

# Text-only LLM prompts (for analyzing TEXT descriptions, not images)
# Used when classifier_source='llm' - LLM receives aggregated text captions with temporal context
LLM_SEGMENT_TIMELINE_BINARY = """You are analyzing a timeline of activity descriptions to identify distinct segments.

Timeline of observations:
{caption}

Task: Identify continuous activity segments based on temporal patterns, merging adjacent or near-adjacent spans with the same label.

Labels:
- work: Person actively engaged in hands-on tasks (electronics, assembly, tools)
- idle: Person not engaged in tasks, standing/sitting without interaction

CRITICAL INSTRUCTIONS:
1. Output ONLY the segment lines, nothing else - NO preamble, NO explanation, NO thinking
2. Each line must be: [start_time]-[end_time]: label (e.g., 0.50-2.30: work)
3. Use exact timestamps from the timeline; do not invent or approximate times
4. Merge adjacent same-label spans; ensure start_time <= end_time. start_time and end_time MUST be timestamps that appear verbatim in the input; they represent the first and last observation in the segment.
5. Output in chronological order

Example (format and derivation only):

If the sorted input timestamps were:
<t=0.50> work
<t=1.20> work
<t=2.30> work
<t=3.10> idle
<t=3.80> work
<t=5.20> work

Then the output MUST be:
0.50-2.30: work
3.10-3.10: idle
3.80-5.20: work"""

LLM_SEGMENT_TIMELINE_MULTI = """You are analyzing a timeline of activity descriptions to identify distinct segments.

Timeline of observations:
{caption}

Task: Identify continuous activity segments based on temporal patterns, merging adjacent or near-adjacent spans with the same label.

Labels:
- assembling_drone: Working with tools, drone parts, wires, screws, assembly tasks
- using_phone: Holding or interacting with a phone
- idle: Standing/sitting without doing any task, arms resting
- unknown: Activity cannot be confidently identified

CRITICAL INSTRUCTIONS:
1. Output ONLY the segment lines, nothing else - NO preamble, NO explanation, NO thinking
2. Each line must be: [start_time]-[end_time]: label (e.g., 0.50-2.30: assembling_drone)
3. Use exact timestamps from the timeline; do not invent or approximate times
4. Merge adjacent same-label spans; ensure start_time < end_time
5. Output in chronological order

Output format (ONLY these lines):
0.50-2.30: assembling_drone
3.10-3.10: using_phone
3.80-5.20: assembling_drone"""




# # Prompt for duration estimation: ask the LLM to return a single integer (seconds)
# DURATION_PROMPT_TEMPLATE = (
#     "You are an assistant that estimates how long a described manual task typically takes.\n"
#     "Given the task description below, respond with a single integer representing the estimated time in seconds.\n"
#     "Do NOT add any explanation or text — only the integer number of seconds.\n\n"
#     "Task: {task}\n"
# )

# Prompt for task completion evaluation
TASK_COMPLETION_PROMPT_TEMPLATE = (
    "You are a task completion evaluator. Based on the captions from video analysis, determine if the specified task has been completed.\n"
    "Respond with exactly one word: either 'yes' or 'no'.\n"
    "Do NOT add any punctuation, explanation, quotes, or extra text — only the single word.\n\n"
    "Task: {task}\nCaptions: {captions}\n"
)


_local_text_llm = None
def get_local_text_llm():
    global _local_text_llm
    if _local_text_llm is not None:
        return _local_text_llm

    if shutil.which('ollama'):
        ollama_model = 'qwen3:0.6b'
        class OllamaWrapper:
            def __init__(self, model_id):
                self.model_id = model_id
            def __call__(self, prompt, max_new_tokens=128):
                try:
                    p = subprocess.run(['ollama', 'run', self.model_id, prompt], capture_output=True, timeout=120)
                    out_bytes = p.stdout or b''
                    try:
                        out = out_bytes.decode('utf-8', errors='replace').strip()
                        print('Ollama decoded output: %s', out)
                    except Exception:
                        print('Ollama output decoding failed, using raw bytes string')
                        out = str(out_bytes)
                        print('Ollama raw output string: %s', out)
                    return [{'generated_text': out}]
                except Exception as e:
                    print('Ollama run failed: %s', e)
                    return [{'generated_text': ''}]
        _local_text_llm = OllamaWrapper(ollama_model)
        print('Using Ollama CLI model %s for text LLM', ollama_model)
        return _local_text_llm

    return None

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
LLM_SEGMENT_TIMELINE_BINARY = """
You are analyzing a sequence of timestamped visual-language model (VLM) captions.
Each timestamp represents a single observation frame, not a true time interval.

Task:
Group consecutive observations into continuous activity segments using binary activity labels.

Labels:
- work: Person actively engaged in hands-on tasks (e.g., electronics work, assembly, using tools)
- idle: Person not engaged in any task; standing or sitting without interaction

Segmentation rules:
- Treat each <t=...> entry as one observation.
- Assign exactly one label (work or idle) to each observation.
- If consecutive observations receive the same label, merge them into a single segment.
- The segment start time is the timestamp of the first observation in the merged group.
- The segment end time is the timestamp of the last observation in the merged group.
- Do NOT infer or invent durations beyond the given timestamps.

Output format:
[start_time]-[end_time]: label

Example Input:
<t=0.50> ...VLM captions...
<t=2.30> ...VLM captions...
<t=3.10> ...VLM captions...
<t=5.20> ...VLM captions...

Dummy timestamp based Example Output:
0.50-2.30: work
3.10-3.10: idle
5.20-5.20: work

Instructions:
1. Process observations in chronological order.
2. Detect activity changes based on semantic meaning, not wording differences.
3. Merge consecutive observations with identical labels.
4. Use only the timestamps provided in the input timeline.

Timeline to analyze based on above example format with real timestamps:
{caption}

Answer:
"""

LLM_SEGMENT_TIMELINE_MULTI = """
You are analyzing a sequence of timestamped visual-language model (VLM) captions.
Each timestamp represents a single observation frame, not a true time interval.

Task:
Group consecutive observations into continuous activity segments based on activity consistency.

Labels:
- idle: Standing or sitting without performing a task
- using_phone: Holding or interacting with a phone or camera-like device
- assembling_drone: Working with tools or assembling drone components
- unknown: Activity cannot be confidently identified

Segmentation rules:
- Treat each <t=...> entry as a single observation.
- Assign one label to each observation.
- If consecutive observations receive the same label, merge them into one segment.
- The segment start time is the timestamp of the first observation in the merged group.
- The segment end time is the timestamp of the last observation in the merged group.
- Do NOT invent new timestamps or durations.

Output format:
[start_time]-[end_time]: label

Example Input:
<t=8.57> ...VLM captions...
<t=10.70> ...VLM captions...
<t=12.83> ...VLM captions...
<t=15.00> ...VLM captions...

Dummy timestamp based Example Output:
8.57-15.00: idle

Instructions:
1. Process the observations in chronological order.
2. Detect activity changes based on semantics, not wording differences.
3. Merge consecutive observations with the same label.
4. Use only the timestamps provided in the input.

Timeline to analyze based on above example format with real timestamps:
{caption}

Answer:
"""




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

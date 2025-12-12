import shutil
import subprocess
import logging

# Prompt template for LLM classification of captions (work vs idle).
CLASSIFY_PROMPT_TEMPLATE = (
    "You are a strict classifier. Respond with exactly one lowercase word: either work or idle.\n"
    "Do NOT add any punctuation, explanation, quotes, or extra text — only the single word.\n\n"
    "Context: {prompt}\nCaption: {caption}\n"
)

# Prompt for duration estimation: ask the LLM to return a single integer (seconds)
DURATION_PROMPT_TEMPLATE = (
    "You are an assistant that estimates how long a described manual task typically takes.\n"
    "Given the task description below, respond with a single integer representing the estimated time in seconds.\n"
    "Do NOT add any explanation or text — only the integer number of seconds.\n\n"
    "Task: {task}\n"
)

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
                        logging.info('Ollama decoded output: %s', out)
                    except Exception:
                        logging.info('Ollama output decoding failed, using raw bytes string')
                        out = str(out_bytes)
                        logging.info('Ollama raw output string: %s', out)
                    return [{'generated_text': out}]
                except Exception as e:
                    logging.info('Ollama run failed: %s', e)
                    return [{'generated_text': ''}]
        _local_text_llm = OllamaWrapper(ollama_model)
        logging.info('Using Ollama CLI model %s for text LLM', ollama_model)
        return _local_text_llm

    return None

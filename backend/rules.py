import re
import logging
from typing import Optional, Tuple

# Centralized rules for determining activity label from caption or LLM output

# Keyword list used to detect 'work' from captions when LLM is not used or inconclusive
WORK_KEYWORDS = [
    'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
    'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
    'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
]

# Named rule sets so the frontend can offer choices.
RULE_SETS = {
    'default': {
        'description': 'Default keyword-based detection',
        'work_keywords': WORK_KEYWORDS,
    },
    'strict': {
        'description': 'Strict matching (fewer keywords)',
        'work_keywords': ['assemble', 'screw', 'weld', 'repair', 'install'],
    },
}


def _extract_text_from_llm_output(cls_out) -> str:
    """Normalize various LLM return shapes into a single string."""
    if cls_out is None:
        return ''
    try:
        if isinstance(cls_out, list) and len(cls_out) > 0:
            f0 = cls_out[0]
            if isinstance(f0, dict):
                return f0.get('generated_text') or f0.get('text') or str(f0)
            return str(f0)
        return str(cls_out)
    except Exception:
        return str(cls_out)


def determine_label(
    caption: str,
    use_llm: bool = False,
    text_llm: Optional[callable] = None,
    prompt: str = '',
    classify_prompt_template: Optional[str] = None,
    max_new_tokens: int = 8,
    rule_set: str = 'default',
    output_mode: str = 'binary',  # 'binary' or 'multi'
) -> Tuple[str, Optional[str]]:
    """Determine `label` ('work' or 'idle') for a given caption.

    Returns a tuple `(label, llm_text)` where `llm_text` is the normalized
    LLM output (or `None` if none was produced).
    """
    label = 'idle'
    cls_text = None

    if use_llm and text_llm is not None and classify_prompt_template:
        try:
            cls_prompt = classify_prompt_template.format(prompt=prompt, caption=caption)
            cls_out = text_llm(cls_prompt, max_new_tokens=max_new_tokens)
            cls_text = _extract_text_from_llm_output(cls_out)
            if cls_text:
                cleaned = re.sub(r'[^\n\w\s]', '', cls_text).strip()
                words = cleaned.split()
                if words:
                    last_word = words[-1].lower()
                    # direct mapping
                    if last_word == 'work':
                        label = 'work'
                    elif last_word == 'idle':
                        label = 'idle'
                    else:
                        # handle short domain labels (e.g. assembling_drone, using_phone)
                        # If caller requested multi-class output, preserve raw label
                        if output_mode == 'multi':
                            label = last_word
                        else:
                            if 'assembl' in last_word or 'drone' in last_word or 'screw' in last_word:
                                label = 'work'
                            elif 'phone' in last_word:
                                label = 'idle'
        except Exception as e:
            logging.info('LLM classification failed: %s', e)

    # Fallback keyword-based detection
    if label == 'idle':
        lw = caption.lower() if isinstance(caption, str) else ''
        kws = RULE_SETS.get(rule_set, {}).get('work_keywords', WORK_KEYWORDS)
        if any(w in lw for w in kws):
            label = 'work'

    return label, cls_text


def list_rule_sets():
    """Return metadata for available rule sets."""
    out = {}
    for name, cfg in RULE_SETS.items():
        out[name] = {'description': cfg.get('description', ''), 'keywords': cfg.get('work_keywords', [])}
    return out


# Classifier prompt templates the frontend can present as choices. These are
# defaults and can be overridden by passing `classifier_prompt` from the UI.
CLASSIFIER_PROMPTS = {
    'blip_binary': (
        "Classify the following caption as exactly one of: 'work' or 'idle'.\n"
        "Only output the single word 'work' or 'idle'.\n\nCaption: {caption}\nAnswer:"
    ),
    'qwen_activity': (
        "You are an expert activity recognition model.\n"
        "Look ONLY at the MAIN PERSON in the image. Choose one label: assembling_drone, idle, using_phone, or unknown.\n"
        "Answer with exactly one label and nothing else.\n\nCaption: {caption}\nAnswer:"
    ),
}


def list_classifiers():
    """Return available classifiers and their default prompt templates."""
    out = {}
    for k, v in CLASSIFIER_PROMPTS.items():
        out[k] = {'prompt_template': v}
    return out
import re
import logging
from typing import Optional, Tuple

# Centralized rules for determining activity label from caption or LLM output

# Keyword list used to detect 'work' from captions when LLM is not used or inconclusive
WORK_KEYWORDS = [
    'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
    'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
    'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
]


def _extract_text_from_llm_output(cls_out) -> str:
    """Normalize various LLM return shapes into a single string."""
    if cls_out is None:
        return ''
    try:
        if isinstance(cls_out, list) and len(cls_out) > 0:
            f0 = cls_out[0]
            if isinstance(f0, dict):
                return f0.get('generated_text') or f0.get('text') or str(f0)
            return str(f0)
        return str(cls_out)
    except Exception:
        return str(cls_out)


def determine_label(caption: str, use_llm: bool = False, text_llm: Optional[callable] = None, prompt: str = '', classify_prompt_template: Optional[str] = None, max_new_tokens: int = 8) -> Tuple[str, Optional[str]]:
    """Determine `label` ('work' or 'idle') for a given caption.

    Returns a tuple `(label, llm_text)` where `llm_text` is the normalized
    LLM output (or `None` if none was produced).

    The decision order is:
    1. If `use_llm` and a `text_llm` callable and `classify_prompt_template` are provided,
       call the LLM and inspect the last cleaned word for 'work' or 'idle'.
    2. If label remains 'idle', check the caption for keywords in `WORK_KEYWORDS`.
    """
    label = 'idle'
    cls_text = None

    if use_llm and text_llm is not None and classify_prompt_template:
        try:
            cls_prompt = classify_prompt_template.format(prompt=prompt, caption=caption)
            cls_out = text_llm(cls_prompt, max_new_tokens=max_new_tokens)
            cls_text = _extract_text_from_llm_output(cls_out)
            if cls_text:
                cleaned = re.sub(r'[^\n\w\s]', '', cls_text).strip()
                words = cleaned.split()
                if words:
                    last_word = words[-1].lower()
                    if last_word == 'work':
                        label = 'work'
                    elif last_word == 'idle':
                        label = 'idle'
        except Exception as e:
            logging.info('LLM classification failed: %s', e)

    # Fallback keyword-based detection
    if label == 'idle':
        lw = caption.lower() if isinstance(caption, str) else ''
        if any(w in lw for w in WORK_KEYWORDS):
            label = 'work'

    return label, cls_text

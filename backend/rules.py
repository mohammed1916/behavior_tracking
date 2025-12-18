"""Rules helpers for converting captions / LLM outputs into activity labels.

This module centralises keyword-based heuristics and small helpers for
normalising LLM outputs used by the backend streaming endpoints.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# Keyword list used to detect 'work' from captions when LLM is not used or inconclusive
WORK_KEYWORDS: List[str] = [
    'make', 'making', 'assemble', 'assembling', 'work', 'working', 'hold', 'holding',
    'use', 'using', 'cut', 'screw', 'weld', 'attach', 'insert', 'paint', 'press',
    'turn', 'open', 'close', 'pick', 'place', 'operate', 'repair', 'install', 'build'
]


# Named rule sets so the frontend can offer choices.
RULE_SETS: Dict[str, Dict[str, Any]] = {
    'default': {
        'description': 'Default keyword-based detection',
        'work_keywords': WORK_KEYWORDS,
    },
}

# Import shared prompt templates for VLM/label instructions
from . import llm as llm_mod

# Label prompt templates (binary vs multi) referenced by "classifier mode"
LABEL_PROMPTS: Dict[str, str] = {
    'binary': llm_mod.LABLE_PROMPT_TEMPLATE_BINARY,
    'multi': llm_mod.LABLE_PROMPT_TEMPLATE_MULTI,
}

# Expected label sets for validation
LABEL_SETS: Dict[str, List[str]] = {
    'binary': ['work', 'idle'],
    'multi': ['assembling_drone', 'idle', 'using_phone', 'unknown'],
}


def _extract_text_from_llm_output(cls_out: Any) -> str:
    """Normalize various LLM return shapes into a single string.

    Supports strings, lists, or a list of dicts with keys like
    'generated_text' or 'text'. Returns empty string for None.
    """
    if cls_out is None:
        return ''
    try:
        if isinstance(cls_out, list) and cls_out:
            first = cls_out[0]
            if isinstance(first, dict):
                return first.get('generated_text') or first.get('text') or str(first)
            return str(first)
        return str(cls_out)
    except Exception:
        # Best-effort fallback
        return str(cls_out)


CLASSIFIER_PROMPTS: Dict[str, str] = {}


def determine_label(
    caption: str,
    use_llm: bool = False,
    text_llm: Optional[Callable[..., Any]] = None,
    prompt: str = '',
    classify_prompt_template: Optional[str] = None,
    max_new_tokens: int = 8,
    rule_set: str = 'default',
    output_mode: str = 'binary',  # 'binary' or 'multi'
) -> Tuple[str, Optional[str]]:
    """Return a label and optional LLM text for a caption.

    Behaviour:
    - If `use_llm` is True and both `text_llm` and `classify_prompt_template`
      are provided, calls the LLM with the rendered prompt and normalises
      the returned text.
    - If the LLM returns a clear 'work' or 'idle' token, that is used.
    - If `output_mode == 'multi'`, unknown short domain labels are preserved.
    - Otherwise falls back to keyword matching using the configured `rule_set`.

    Returns:
        (label, llm_text) where `llm_text` is the raw-normalised LLM output
        or `None` if no LLM output was produced.
    """
    label: str = 'idle'
    cls_text: Optional[str] = None

    # Build the final prompt: prefer an explicit classify_prompt_template
    # passed by caller; otherwise compose from shared VLM + label + rules templates
    if use_llm and text_llm is not None:
        # try:
        if classify_prompt_template:
            final_template = classify_prompt_template
        else:
            label_tpl = LABEL_PROMPTS.get(output_mode, LABEL_PROMPTS['binary'])
            final_template = llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + label_tpl + "\n" + llm_mod.RULES_PROMPT_TEMPLATE
        rendered = final_template.format(prompt=prompt, caption=caption)
        cls_out = text_llm(rendered, max_new_tokens=max_new_tokens)
        cls_text = _extract_text_from_llm_output(cls_out) or None
        if cls_text:
            cleaned = re.sub(r'[^\n\w\s]', '', cls_text).strip()
            tokens = [t for t in cleaned.split() if t]
            if tokens:
                last = tokens[-1].lower()
                expected = LABEL_SETS.get(output_mode, LABEL_SETS['binary'])
                if last in expected:
                    label = last
                else:
                    # # If caller requested binary output but model returned a multi label,
                    # # deterministically map multi->binary: non-idle -> work
                    # if output_mode == 'binary':
                    #     if any(sub in last for sub in ('assemble', 'drone', 'screw', 'install', 'repair', 'work')):
                    #         label = 'work'
                    #     elif 'phone' in last:
                    #         label = 'idle'
                    # else:
                    # preserve unknown / try to coerce to a known multi label
                    if any(sub in last for sub in ('assemble', 'drone')):
                        label = 'assembling_drone'
                    elif 'phone' in last:
                        label = 'using_phone'
                    elif 'idle' in last:
                        label = 'idle'
                    else:
                        label = 'unknown'
        # except Exception:
        #     logger.exception('LLM classification failed')

    # # Keyword fallback when still 'idle'
    # if label == 'idle':
    
    if not use_llm or cls_text is None:
        lw = caption.lower() if isinstance(caption, str) else ''
        kws = RULE_SETS.get(rule_set, {}).get('work_keywords', WORK_KEYWORDS)
        if any(k in lw for k in kws):
            label = 'work' # if not label is alredy initialized as idle

    return label, cls_text


def list_rule_sets() -> Dict[str, Dict[str, Any]]:
    """Return metadata for available rule sets.

    The returned mapping contains { name: { 'description': ..., 'keywords': [...] } }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in RULE_SETS.items():
        out[name] = {
            'description': cfg.get('description', ''),
            'keywords': list(cfg.get('work_keywords', [])),
        }
    return out


def list_label_modes() -> Dict[str, Dict[str, Any]]:
    """Return available classifier modes (label templates and labels)."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, tpl in LABEL_PROMPTS.items():
        out[name] = {'prompt_template': tpl, 'labels': list(LABEL_SETS.get(name, []))}
    return out

def get_label_template(mode: str) -> Optional[str]:
    return LABEL_PROMPTS.get(mode)


__all__ = [
    'determine_label',
    'list_rule_sets',
    'list_label_modes',
    'get_label_template',
    'CLASSIFIER_PROMPTS',
    'RULE_SETS',
    'WORK_KEYWORDS',
]

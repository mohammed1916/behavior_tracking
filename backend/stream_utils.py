import base64
from typing import Optional, List, Dict, Any, Callable
import inspect

import backend.llm as llm_mod
import backend.rules as rules_mod


def normalize_caption_output(captioner, output: Any) -> str:
    """Normalize captioner output to a single caption string.

    Handles various output formats:
    - [{"generated_text": "..."}] (Qwen, transformers pipeline)
    - "plain string"
    - {"caption": "..."}
    """
    import json
    
    if not output:
        return ""

    # Handle list format (common from transformers pipelines)
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            text = first.get('generated_text') or first.get('caption') or ""
            return text.strip() if text else ""
        return str(first).strip()

    # Handle dict format
    if isinstance(output, dict):
        text = output.get('generated_text') or output.get('caption') or ""
        return text.strip() if text else ""

    # Handle string format (including stringified dicts)
    output_str = str(output).strip()
    
    # Try to parse if it looks like a stringified dict
    if output_str.startswith('{') and output_str.endswith('}'):
        try:
            parsed = json.loads(output_str)
            if isinstance(parsed, dict):
                text = parsed.get('generated_text') or parsed.get('caption') or ""
                return text.strip() if text else ""
        except (json.JSONDecodeError, ValueError):
            pass

    return output_str


def call_captioner(captioner, img, prompt: Optional[str], on_debug: Optional[Callable[[str], None]] = None):
    """Call captioner with optional prompt and debug callback, passing only supported kwargs.

    Uses inspect.signature to detect accepted parameters to avoid relying on nested TypeError fallbacks.
    """
    eff_prompt = None if (prompt is None or str(prompt).strip() == "") else prompt

    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(captioner)
        params = sig.parameters
    except (TypeError, ValueError):
        # If we cannot introspect, attempt a simple direct call as fallback
        params = {}

    if eff_prompt is not None and 'prompt' in params:
        kwargs['prompt'] = eff_prompt
    if on_debug is not None and 'on_debug' in params:
        kwargs['on_debug'] = on_debug

    try:
        return captioner(img, **kwargs)
    except TypeError:
        # Final fallback in case of unexpected signature quirks
        return captioner(img)


def build_vlm_prompt_for_source(classifier_source_norm: str, classifier_mode: str, classifier_prompt: Optional[str]) -> Optional[str]:
    """Return the prompt sent to the VLM for the given classifier source.
    
    Returns a non-empty prompt for image captioning/description, or None to use model defaults.
    """
    if classifier_source_norm == 'vlm':
        label_tpl = llm_mod.LABLE_PROMPT_TEMPLATE_MULTI if classifier_mode == 'multi' else llm_mod.LABLE_PROMPT_TEMPLATE_BINARY
        prompt = classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + label_tpl + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
        return prompt if prompt and prompt.strip() else None
    
    if classifier_source_norm == 'bow':
        prompt = classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
        return prompt if prompt and prompt.strip() else None
    
    # 'llm' mode: ask for plain description; let model use its base template
    if classifier_prompt:
        return classifier_prompt if classifier_prompt.strip() else None
    
    # Return a simple description task prompt
    return llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\nDescribe the visible activities and actions of the main person in detail."


def build_classify_prompt_template(classifier_source_norm: str, classifier_mode: str, classifier_prompt: Optional[str]) -> Optional[str]:
    """Return classify prompt template used for downstream LLM label decisions."""
    if classifier_source_norm == 'vlm':
        label_tpl = llm_mod.LABLE_PROMPT_TEMPLATE_MULTI if classifier_mode == 'multi' else llm_mod.LABLE_PROMPT_TEMPLATE_BINARY
        return classifier_prompt or (llm_mod.VLM_BASE_PROMPT_TEMPLATE + "\n" + label_tpl + "\n" + llm_mod.RULES_PROMPT_TEMPLATE)
    if classifier_source_norm == 'bow':
        return None
    return classifier_prompt or rules_mod.get_label_template(classifier_mode)


def per_sample_label_for_source(
    classifier_source_norm: str,
    classifier_mode: str,
    caption: str,
    prompt: str,
    rule_set: str,
    classify_prompt_template: Optional[str],
    effective_use_llm: bool,
):
    """Compute per-sample label + text. For 'llm' source we defer labeling to windowed LLM (returns None)."""
    if classifier_source_norm == 'vlm':
        label = rules_mod.normalize_label_text(caption, output_mode=classifier_mode)
        return label, caption
    if classifier_source_norm == 'bow':
        output_mode = 'multi' if classifier_mode == 'label' else classifier_mode
        label, cls_text = rules_mod.determine_label(
            caption,
            use_llm=False,
            text_llm=None,
            prompt=prompt,
            classify_prompt_template=classify_prompt_template,
            rule_set=rule_set,
            output_mode=output_mode,
        )
        return label, cls_text
    return None, None


def merge_and_filter_ranges(ranges: List[Dict[str, Any]], min_segment_sec: float, merge_gap_sec: float) -> List[Dict[str, Any]]:
    """Merge nearby ranges and drop very short segments.

    ranges: list of dicts with 'startTime' and 'endTime'
    Returns a new list of merged, filtered ranges.
    """
    if not ranges:
        return []
    rs = sorted(ranges, key=lambda r: r.get('startTime', 0))
    merged: List[Dict[str, Any]] = []
    cur = rs[0].copy()
    for r in rs[1:]:
        gap = r.get('startTime', 0) - cur.get('endTime', 0)
        if gap <= merge_gap_sec:
            cur['endTime'] = max(cur.get('endTime', 0), r.get('endTime', 0))
            cur['endFrame'] = max(cur.get('endFrame', cur.get('endFrame', 0)), r.get('endFrame', r.get('endFrame', 0)))
        else:
            dur = cur.get('endTime', 0) - cur.get('startTime', 0)
            if dur >= min_segment_sec:
                merged.append(cur.copy())
            cur = r.copy()
    dur = cur.get('endTime', 0) - cur.get('startTime', 0)
    if dur >= min_segment_sec:
        merged.append(cur.copy())
    return merged

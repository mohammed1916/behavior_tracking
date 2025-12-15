import sys
import types
import pytest


@pytest.fixture(autouse=True)
def inject_llm_module(monkeypatch):
    """Auto-inject a minimal `llm` module for tests that expect it.

    This provides `TASK_COMPLETION_PROMPT_TEMPLATE` and a `get_local_text_llm`
    callable that returns a simple dummy LLM. Tests can still pass their own
    `llm` callable to functions that accept it.
    """
    if 'llm' in sys.modules:
        # already present (real or test-provided) — don't override
        yield
        return

    m = types.ModuleType('llm')
    m.TASK_COMPLETION_PROMPT_TEMPLATE = "Task: {task}\nCaptions:\n{captions}\nDid the task complete?"

    def get_local_text_llm():
        def _dummy(prompt):
            return [{'generated_text': 'uncertain'}]
        return _dummy

    m.get_local_text_llm = get_local_text_llm
    monkeypatch.setitem(sys.modules, 'llm', m)
    yield
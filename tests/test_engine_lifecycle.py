"""Exit must reclaim cyclic owners even when automatic GC is disabled."""
import gc
import weakref

import pytest


def test_exit_collects_runner_cycles_and_is_idempotent(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from nanovllm.engine.llm_engine import LLMEngine

    calls = []

    class CyclicRunner:
        def __init__(self):
            self.callback = self.call

        def call(self, method):
            calls.append(method)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    engine = LLMEngine.__new__(LLMEngine)
    engine.ps = []
    engine.model_runner = CyclicRunner()
    runner_ref = weakref.ref(engine.model_runner)
    enabled = gc.isenabled()
    gc.disable()
    try:
        engine.exit()
        assert runner_ref() is None
        engine.exit()
        assert calls == ["exit", "empty_cache"]
    finally:
        if enabled:
            gc.enable()

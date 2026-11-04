"""Explicit opt-in real-model gate. Run before collecting performance claims."""
import os
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("reuse", [False, True])
def test_real_model_greedy_equivalence(eager, reuse):
    model = os.environ.get("NANO_MODEL")
    if not model:
        pytest.skip("set NANO_MODEL to a local Qwen3 or dense Llama checkpoint on a CUDA host")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.policy import SchedulingOptions
    prompts = ["Explain KV caching in one sentence.", "The capital of France is", "Explain attention. " * 90]
    outputs = {}
    initial_allocated = torch.cuda.memory_allocated()
    for mode in ["native", "fixed", "adaptive"]:
        engine = LLM(model, scheduling=SchedulingOptions(mode=mode, prefill_budget=32),
                     dtype=os.environ.get("NANO_DTYPE", "auto"),
                     max_num_seqs=8, max_num_batched_tokens=256, max_model_len=4096,
                     enforce_eager=eager, kv_cache_block_cap=64, reuse_decode_buffers=reuse, profile_stages=True)
        try:
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
            # Staggered arrivals force mixed prefill/decode for modified modes.
            seqs = [engine.add_request(prompts[0], params)]
            for _ in range(3):
                engine.step()
            seqs.extend(engine.add_request(p, params) for p in prompts[1:])
            while not engine.is_finished():
                engine.step()
            outputs[mode] = [seq.completion_token_ids for seq in seqs]
            assert not engine.scheduler.block_manager.used_block_ids
            records = engine.model_runner.profiler.collect(wait=True)
            assert records and all(v >= 0 for r in records for v in r["cuda_stream_ms"].values())
        finally:
            engine.exit()
        # Permit small shared kernel/RoPE caches, but never a retained model.
        assert torch.cuda.memory_allocated() < initial_allocated + 64 * 1024**2
    assert outputs["native"] == outputs["fixed"] == outputs["adaptive"]


@pytest.mark.gpu
def test_initialization_failure_restores_process_state(monkeypatch):
    model = os.environ.get("NANO_MODEL")
    if not model:
        pytest.skip("set NANO_MODEL on a CUDA host")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import get_context
    dtype, device = torch.get_default_dtype(), torch.get_default_device()

    def fail_warmup(self):
        raise RuntimeError("injected warmup failure")

    monkeypatch.setattr(ModelRunner, "warmup_model", fail_warmup)
    with pytest.raises(RuntimeError, match="injected warmup failure"):
        ModelRunner(Config(model, max_num_seqs=8, max_num_batched_tokens=256,
                           enforce_eager=True, kv_cache_block_cap=64), 0, [])
    assert not torch.distributed.is_initialized()
    assert torch.get_default_dtype() == dtype
    assert torch.get_default_device() == device
    assert get_context().slot_mapping is None

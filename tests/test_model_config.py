"""Configuration and stop semantics tests; no torch/CUDA dependency."""
import json
from types import SimpleNamespace
import pytest
from nanovllm.models.registry import validate_model_config
from nanovllm.models.rope_config import validate_rope_scaling
from nanovllm.utils.generation import resolve_eos_token_ids
from benchmarks.simulator import SimEngine
from nanovllm import SamplingParams
from nanovllm.engine.policy import SchedulingOptions


def config(**kwargs):
    values = dict(model_type="llama", architectures=["LlamaForCausalLM"], hidden_act="silu",
                  num_attention_heads=4, num_key_value_heads=2, hidden_size=64)
    return SimpleNamespace(**(values | kwargs))


@pytest.mark.parametrize("updates,match", [
    ({"model_type": "mllama"}, "model_type"),
    ({"architectures": ["LlamaForSequenceClassification"]}, "architectures"),
    ({"quantization_config": {"quant_method": "awq"}}, "Quantized"),
    ({"hidden_act": "gelu"}, "silu"),
    ({"num_key_value_heads": 3}, "heads"),
    ({"partial_rotary_factor": .5}, "Partial"),
    ({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, "RoPE type"),
    ({"sliding_window": 128}, "Sliding"),
])
def test_reject_unsupported_architecture_features(updates, match):
    with pytest.raises(ValueError, match=match):
        validate_model_config(config(**updates))


@pytest.mark.parametrize("scaling", [
    {"rope_type": "linear", "factor": float("nan")},
    {"rope_type": "linear", "factor": 0},
    {"rope_type": "default", "factor": 4},
    {"rope_type": "llama3", "factor": 8, "low_freq_factor": 4, "high_freq_factor": 1,
     "original_max_position_embeddings": 8192},
])
def test_reject_invalid_rope(scaling):
    with pytest.raises(ValueError, match="RoPE"):
        validate_rope_scaling(scaling)


def test_eos_precedence_and_validation(tmp_path):
    hf = SimpleNamespace(eos_token_id=2, vocab_size=100)
    tokenizer = SimpleNamespace(eos_token_id=3)
    assert resolve_eos_token_ids(tmp_path, hf, tokenizer) == (2,)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 5, 5]}))
    assert resolve_eos_token_ids(tmp_path, hf, tokenizer) == (2, 5)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 100}')
    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_eos_token_ids(tmp_path, hf, tokenizer)
    (tmp_path / "generation_config.json").unlink()
    hf.eos_token_id = None
    assert resolve_eos_token_ids(tmp_path, hf, tokenizer) == (3,)
    tokenizer.eos_token_id = None
    assert resolve_eos_token_ids(tmp_path, hf, tokenizer) == ()


@pytest.mark.parametrize("mode", ["native", "fixed", "adaptive"])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_multiple_eos_stop_and_free_cache(mode, ignore_eos):
    engine = SimEngine(SchedulingOptions(mode=mode), blocks=16, block_size=4)
    engine.config.eos = [77, 78]
    from nanovllm.engine.adaptive_scheduler import make_scheduler
    engine.scheduler = make_scheduler(engine.config)
    seq = engine.add_request([1, 2], SamplingParams(max_tokens=3, ignore_eos=ignore_eos))
    selected, is_prefill = engine.scheduler.schedule()
    engine.scheduler.postprocess(selected, [78], is_prefill)
    assert seq.is_finished == (not ignore_eos)
    if not ignore_eos:
        assert not engine.scheduler.block_manager.used_block_ids

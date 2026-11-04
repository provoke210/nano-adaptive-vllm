"""Explicit model dispatch; importing this module does not import CUDA kernels."""

_ARCHITECTURES = {"qwen3": "Qwen3ForCausalLM", "llama": "LlamaForCausalLM"}


def validate_model_config(config):
    model_type = getattr(config, "model_type", None)
    if model_type not in _ARCHITECTURES:
        raise ValueError(f"Unsupported model_type {model_type!r}; supported: qwen3, llama (dense text only)")
    architectures = getattr(config, "architectures", None)
    if architectures and architectures != [_ARCHITECTURES[model_type]]:
        raise ValueError(f"Unsupported architectures {architectures!r} for {model_type}")
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Quantized checkpoints are not supported; use floating-point safetensors")
    if getattr(config, "hidden_act", None) != "silu":
        raise ValueError("Only silu gated MLP is supported")
    if getattr(config, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("Partial rotary embeddings are not supported")
    if getattr(config, "sliding_window", None) and getattr(config, "use_sliding_window", True):
        raise ValueError("Sliding-window attention is not supported")
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    if heads < 1 or kv_heads < 1:
        raise ValueError("Require positive attention heads and KV heads")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    if heads < 1 or kv_heads < 1 or heads % kv_heads or head_dim < 2 or head_dim % 2:
        raise ValueError("Require positive heads, heads divisible by KV heads, and an even head dimension")
    # Do not silently run an unimplemented scaling method as ordinary RoPE.
    from nanovllm.models.rope_config import validate_rope_scaling
    validate_rope_scaling(getattr(config, "rope_scaling", None))
    return model_type


def get_model_class(config):
    model_type = validate_model_config(config)
    if model_type == "llama":
        from nanovllm.models.llama import LlamaForCausalLM
        return LlamaForCausalLM
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    return Qwen3ForCausalLM

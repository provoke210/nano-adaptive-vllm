"""Validation for the static RoPE variants implemented by this fork."""
import math


def validate_rope_scaling(scaling):
    if scaling is None:
        return "default"
    if not isinstance(scaling, dict):
        raise ValueError("rope_scaling must be a dictionary")
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if "type" in scaling and "rope_type" in scaling and scaling["type"] != scaling["rope_type"]:
        raise ValueError("Conflicting RoPE type and rope_type")
    if kind not in {"default", "linear", "llama3"}:
        raise ValueError(f"Unsupported RoPE type {kind!r}; supported: default, linear, llama3")
    if kind == "default":
        if set(scaling) - {"type", "rope_type", "rope_theta"}:
            raise ValueError("Default RoPE must not contain scaling parameters")
        return kind
    allowed = {"type", "rope_type", "factor"}
    if kind == "llama3":
        allowed |= {"low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
    if set(scaling) - allowed:
        raise ValueError(f"Unsupported RoPE options: {set(scaling) - allowed}")
    for key in allowed - {"type", "rope_type"}:
        value = scaling.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"RoPE {key} must be finite and positive")
    if scaling["factor"] < 1:
        raise ValueError("RoPE factor must be >= 1")
    if kind == "llama3" and scaling["high_freq_factor"] <= scaling["low_freq_factor"]:
        raise ValueError("RoPE high_freq_factor must exceed low_freq_factor")
    return kind

"""Read only stop tokens; sampling parameters remain controlled by the caller."""
import json
from pathlib import Path


def resolve_eos_token_ids(model_path, hf_config, tokenizer):
    path = Path(model_path) / "generation_config.json"
    settings = json.loads(path.read_text()) if path.is_file() else {}
    value = settings.get("eos_token_id")
    if value is None:
        value = getattr(hf_config, "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else [value]
    if not all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < hf_config.vocab_size for token in values):
        raise ValueError("eos_token_id must contain valid vocabulary token IDs")
    return tuple(dict.fromkeys(values))

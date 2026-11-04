"""Strict safetensors loading, including packed projections and tied embeddings."""
import json
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in names):
            raise ValueError("Checkpoint shard names must be local file names")
        files = [root / name for name in names]
    else:
        weight_map = None
        files = sorted(root.glob("*.safetensors"))
    if not files or any(not file.is_file() for file in files):
        raise ValueError("Missing safetensors checkpoint or shard")
    mapping = getattr(model, "packed_modules_mapping", {})
    expected, labels = set(), {}
    for name, param in model.named_parameters(remove_duplicate=False):
        module_name = name.split(".")[-2]
        shards = [shard for packed, shard in mapping.values() if packed == module_name] or [None]
        for shard in shards:
            key = (id(param), shard)
            expected.add(key)
            labels[key] = f"{name} [{shard}]" if shard is not None else name
    loaded, seen_names = set(), set()
    for file in files:
        with safe_open(str(file), "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name in seen_names:
                    raise ValueError(f"Duplicate checkpoint tensor: {weight_name}")
                if weight_map is not None and weight_map.get(weight_name) != file.name:
                    raise ValueError(f"Tensor absent from index or in wrong shard: {weight_name}")
                seen_names.add(weight_name)
                parts = weight_name.split(".")
                shard = None
                if len(parts) > 1 and parts[-2] in mapping:
                    parts[-2], shard = mapping[parts[-2]]
                param_name = ".".join(parts)
                try:
                    param = model.get_parameter(param_name)
                except AttributeError as exc:
                    raise ValueError(f"Unexpected checkpoint tensor: {weight_name}") from exc
                tensor = f.get_tensor(weight_name)
                if not tensor.is_floating_point():
                    raise ValueError(f"Expected floating-point weight: {weight_name}")
                module = model.get_submodule(".".join(parts[:-1]))
                shape = list(param.shape)
                dim = getattr(module, "tp_dim", None)
                if hasattr(module, "num_embeddings_per_partition"):
                    dim = 0
                if param.ndim == 1 and dim == 1:  # RowParallelLinear bias is replicated.
                    dim = None
                if dim is not None:
                    shape[dim] *= module.tp_size
                if shard is not None:
                    if isinstance(shard, str):
                        heads = module.num_heads if shard == "q" else module.num_kv_heads
                        shape[0] = heads * module.head_size * module.tp_size
                    else:
                        shape[0] = module.output_sizes[shard]
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError(f"Shape mismatch for {weight_name}: expected {tuple(shape)}, got {tuple(tensor.shape)}")
                key = (id(param), shard)
                if key not in expected:
                    raise ValueError(f"Packed weights must use original projection names: {weight_name}")
                if key in loaded:
                    # Tied checkpoints may serialize both aliases, but not conflicting values.
                    if shard is not None or not torch.equal(param.detach().cpu(), tensor.to(param.dtype)):
                        raise ValueError(f"Conflicting tied or duplicate weight: {weight_name}")
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard)
                loaded.add(key)
    if weight_map is not None and seen_names != set(weight_map):
        raise ValueError(f"Missing indexed tensors: {sorted(set(weight_map) - seen_names)}")
    missing = expected - loaded
    if missing:
        raise ValueError(f"Missing checkpoint weights: {sorted(labels[key] for key in missing)}")

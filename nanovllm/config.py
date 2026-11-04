import os
from dataclasses import dataclass, field
from transformers import AutoConfig
from nanovllm.engine.policy import SchedulingOptions
from nanovllm.models.registry import validate_model_config


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    dtype: str = "auto"
    batch_invariant: bool = True
    reuse_decode_buffers: bool = False
    profile_stages: bool = False
    hf_config: AutoConfig | None = None
    eos: int | list[int] | tuple[int, ...] = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_block_cap: int | None = None
    scheduling: SchedulingOptions = field(default_factory=SchedulingOptions)

    def __post_init__(self):
        if self.dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float16, or bfloat16")
        if self.tensor_parallel_size != 1:
            raise ValueError("This experimental fork currently validates single-GPU execution only")
        if self.max_num_batched_tokens < self.max_num_seqs or self.max_num_seqs < 1:
            raise ValueError("Require token budget >= max_num_seqs >= 1")
        if self.kv_cache_block_cap is not None and self.kv_cache_block_cap < 1:
            raise ValueError("kv_cache_block_cap must be positive")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        validate_model_config(self.hf_config)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

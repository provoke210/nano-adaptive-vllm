"""Offline numerical integration gates using random HF models, not pretrained weights.

The real model layers, packed loader, RoPE and logits run on CPU. Only CUDA
attention is replaced with the explicit reference below; single-rank metadata
and compiled elementwise wrappers are bypassed. This is not a CUDA kernel test.
"""
import json
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file
from nanovllm.layers.attention import Attention
from nanovllm.layers.rotary_embedding import RotaryEmbedding, get_rope, _get_rope
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.activation import SiluAndMul
from nanovllm.models.registry import get_model_class
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import get_context, set_context, reset_context


def reference_attention(self, q, k, v):
    ctx = get_context()
    if self.k_cache.numel():
        slots = ctx.slot_mapping.long()
        valid = slots >= 0
        self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[slots[valid]] = k[valid]
        self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[slots[valid]] = v[valid]
    if ctx.is_prefill:
        q_ends, k_ends = ctx.cu_seqlens_q.tolist(), ctx.cu_seqlens_k.tolist()
        lengths = [b-a for a, b in zip(k_ends, k_ends[1:])]
    else:
        q_ends = list(range(q.size(0)+1))
        lengths = ctx.context_lens.tolist()
    outputs = []
    for i, length in enumerate(lengths):
        query = q[q_ends[i]:q_ends[i+1]].transpose(0, 1)
        if ctx.block_tables is None:
            key, value = k[k_ends[i]:k_ends[i+1]], v[k_ends[i]:k_ends[i+1]]
        else:
            page_size = self.k_cache.size(1)
            pages = ctx.block_tables[i, :(length + page_size - 1)//page_size].long()
            key = self.k_cache[pages].reshape(-1, self.num_kv_heads, self.head_dim)[:length]
            value = self.v_cache[pages].reshape(-1, self.num_kv_heads, self.head_dim)[:length]
        key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1).transpose(0, 1)
        value = value.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1).transpose(0, 1)
        scores = (query @ key.transpose(-1, -2)) * self.scale
        q_positions = torch.arange(length-query.size(1), length)
        scores.masked_fill_(torch.arange(length)[None, :] > q_positions[:, None], -torch.inf)
        outputs.append((scores.softmax(-1) @ value).transpose(0, 1))
    return torch.cat(outputs)


@pytest.fixture(autouse=True)
def cpu_reference_backend(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(Attention, "forward", reference_attention)
    for cls, name in [(RotaryEmbedding, "forward"), (RMSNorm, "rms_forward"),
                      (RMSNorm, "add_rms_forward"), (SiluAndMul, "forward")]:
        monkeypatch.setattr(cls, name, getattr(cls, name).__wrapped__)
    _get_rope.cache_clear()
    reset_context()
    yield
    reset_context()
    _get_rope.cache_clear()


def make_config(kind="llama", **kwargs):
    cls = transformers.LlamaConfig if kind == "llama" else transformers.Qwen3Config
    return cls(vocab_size=96, hidden_size=64, intermediate_size=112, num_hidden_layers=2,
               num_attention_heads=4, max_position_embeddings=256,
               **({"num_key_value_heads": 2, "head_dim": 16} | kwargs))


def create_pair(tmp_path, config, sharded=False):
    torch.manual_seed(804)
    ref = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    ref.save_pretrained(tmp_path, max_shard_size="10KB" if sharded else "5GB")
    model = get_model_class(config)(config).eval()
    load_model(model, str(tmp_path))
    return ref, model


def prefill(model, tokens):
    length = tokens.numel()
    ends = torch.tensor([0, length], dtype=torch.int32)
    set_context(True, cu_seqlens_q=ends, cu_seqlens_k=ends, max_seqlen_q=length, max_seqlen_k=length)
    hidden = model(tokens, torch.arange(length))
    # Compare every position rather than only the sampled final row.
    get_context().sample_indices = torch.arange(length)
    return model.compute_logits(hidden)


VARIANTS = [
    ("llama", {}),
    ("llama", {"num_key_value_heads": 4}),
    ("llama", {"head_dim": 32}),
    ("llama", {"attention_bias": True, "mlp_bias": True}),
    ("llama", {"tie_word_embeddings": True}),
    ("llama", {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}),
    ("llama", {"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                 "high_freq_factor": 4.0, "original_max_position_embeddings": 32}}),
    ("qwen3", {}),
    ("qwen3", {"tie_word_embeddings": True}),
]


@pytest.mark.parametrize("kind,options", VARIANTS)
@torch.inference_mode()
def test_full_model_logits_and_free_generation(tmp_path, kind, options):
    config = make_config(kind, **options)
    ref, model = create_pair(tmp_path, config, sharded=True)
    tokens = torch.tensor([1, 7, 31, 8, 12, 22, 19])
    for _ in range(4):
        expected = ref(tokens[None], use_cache=False).logits[0]
        actual = prefill(model, tokens)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        assert actual[-1].argmax().item() == expected[-1].argmax().item()
        tokens = torch.cat([tokens, actual[-1].argmax().view(1)])
    if config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight
    if kind == "llama":
        assert not any("q_norm" in name or "k_norm" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize("scaling", [None, {"type": "linear", "factor": 2.0},
    {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
     "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}])
@torch.inference_mode()
def test_rope_high_positions_against_hf(scaling):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
    config = make_config(head_dim=64, rope_scaling=scaling)
    config.max_position_embeddings = 16384
    ref = LlamaRotaryEmbedding(config)
    rope = get_rope(64, 64, 16384, config.rope_theta, config.rope_scaling)
    positions = torch.tensor([0, 1, 8191, 8192, 12001, 16383])
    q, k = torch.randn(6, 4, 64), torch.randn(6, 2, 64)
    cos, sin = ref(q, positions[None])
    expected_q, expected_k = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    actual_q, actual_k = rope(positions, q, k)
    torch.testing.assert_close(actual_q, expected_q[0].transpose(0, 1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_k, expected_k[0].transpose(0, 1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["llama", "qwen3"])
@torch.inference_mode()
def test_paged_chunked_prefill_mixed_batch_and_decode(tmp_path, kind):
    config = make_config(kind)
    ref, model = create_pair(tmp_path, config)
    sequences = [torch.tensor([1, 7, 31, 8, 12, 22, 19]), torch.tensor([3, 14, 5, 6, 23])]
    tables = torch.tensor([[3, 1, 5, 7], [4, 0, 6, 2]], dtype=torch.int32)
    for module in model.modules():
        if isinstance(module, Attention):
            module.k_cache = torch.full((8, 4, module.num_kv_heads, module.head_dim), float("nan"))
            module.v_cache = torch.full_like(module.k_cache, float("nan"))
    lengths = [0, 0]

    def execute(parts, is_prefill):
        inputs, positions, slots, ends = [], [], [], [0]
        for i, part in enumerate(parts):
            pos = torch.arange(lengths[i], lengths[i]+len(part))
            inputs.append(part)
            positions.append(pos)
            slots.extend((tables[i, pos//4].long()*4 + pos%4).tolist())
            lengths[i] += len(part)
            ends.append(ends[-1]+len(part))
        k_ends = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        set_context(is_prefill, cu_seqlens_q=torch.tensor(ends, dtype=torch.int32), cu_seqlens_k=k_ends,
                    max_seqlen_q=max(map(len, parts)), max_seqlen_k=max(lengths),
                    slot_mapping=torch.tensor(slots), context_lens=torch.tensor(lengths), block_tables=tables)
        get_context().sample_indices = torch.tensor(ends[1:])-1
        return model.compute_logits(model(torch.cat(inputs), torch.cat(positions)))

    actual = execute([sequences[0][:3], sequences[1]], True)
    for i, length in enumerate(lengths):
        torch.testing.assert_close(actual[i], ref(sequences[i][:length][None]).logits[0, -1], rtol=3e-5, atol=3e-6)
    # First request still prefills; second now decodes in the same batch.
    next_b = actual[1].argmax().view(1)
    sequences[1] = torch.cat([sequences[1], next_b])
    actual = execute([sequences[0][3:], next_b], True)
    for _ in range(3):
        for i in range(2):
            expected = ref(sequences[i][None], use_cache=False).logits[0, -1]
            torch.testing.assert_close(actual[i], expected, rtol=3e-5, atol=3e-6)
        parts = [row.argmax().view(1) for row in actual]
        sequences = [torch.cat([seq, part]) for seq, part in zip(sequences, parts)]
        actual = execute(parts, False)


@pytest.mark.parametrize("fault", ["missing", "shape", "extra", "missing_shard", "tied_conflict"])
def test_loader_rejects_bad_checkpoints(tmp_path, fault):
    config = make_config(tie_word_embeddings=fault == "tied_conflict")
    ref, model = create_pair(tmp_path, config)
    state = {name: value.clone() for name, value in ref.state_dict().items()}
    if fault == "missing":
        del state["model.layers.0.self_attn.k_proj.weight"]
    elif fault == "shape":
        state["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(1, 64)
    elif fault == "extra":
        state["model.wrong.weight"] = torch.zeros(1)
    elif fault == "tied_conflict":
        state["lm_head.weight"].add_(1)
    else:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        load_model(model, str(tmp_path))


def test_config_dispatch_from_local_hf_files(tmp_path):
    from nanovllm.config import Config
    config = make_config()
    config.save_pretrained(tmp_path)
    loaded = Config(str(tmp_path))
    assert get_model_class(loaded.hf_config).__name__ == "LlamaForCausalLM"
    config.architectures = ["LlamaForSequenceClassification"]
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="architectures"):
        Config(str(tmp_path))


@torch.inference_mode()
def test_rmsnorm_does_not_overwrite_fp32_residual():
    norm = RMSNorm(8)
    x = torch.randn(5, 8)
    original = x.clone()
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
    torch.testing.assert_close(norm(x), expected)
    assert torch.equal(x, original)
    residual = torch.randn_like(x)
    saved = residual.clone()
    normalized, combined = norm(x, residual)
    expected_sum = original + saved
    torch.testing.assert_close(combined, expected_sum)
    torch.testing.assert_close(normalized, expected_sum * torch.rsqrt(expected_sum.square().mean(-1, keepdim=True) + norm.eps))
    assert torch.equal(x, original) and torch.equal(residual, saved)


@pytest.mark.parametrize("tied", [False, True])
@torch.inference_mode()
def test_fp16_llama_logits(tmp_path, tied):
    config = make_config(tie_word_embeddings=tied)
    ref, model = create_pair(tmp_path, config)
    ref.half()
    model.half()
    # Keep RoPE cache FP32, as in ModelRunner's construction under a half default dtype.
    for layer in model.model.layers:
        layer.self_attn.rotary_emb = RotaryEmbedding(16, 16, 256, config.rope_theta)
    tokens = torch.tensor([1, 9, 12, 14, 38, 7, 5])
    torch.testing.assert_close(prefill(model, tokens).float(), ref(tokens[None]).logits[0].float(),
                               rtol=.01, atol=.003)

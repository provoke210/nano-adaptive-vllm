"""Dense Llama text decoder using the engine's packed projections and paged KV.

Architecture reference: Hugging Face Transformers v4.55.2 Llama implementation.
Unlike Qwen3, Llama has no per-head Q/K RMSNorm (even with bias-free QKV).
"""
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        world_size = dist.get_world_size()
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads // world_size
        self.num_kv_heads = config.num_key_value_heads // world_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        bias = getattr(config, "attention_bias", False)
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim,
                                         config.num_attention_heads, config.num_key_value_heads, bias=bias)
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim,
                                        config.hidden_size, bias=bias)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, config.max_position_embeddings,
                                   config.rope_theta, getattr(config, "rope_scaling", None))
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, self.num_kv_heads)

    def forward(self, positions, hidden_states):
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).reshape(-1, self.q_size))


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        bias = getattr(config, "mlp_bias", False)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size, [config.intermediate_size] * 2, bias=bias)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=bias)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

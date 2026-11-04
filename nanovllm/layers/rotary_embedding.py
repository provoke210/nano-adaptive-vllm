from functools import lru_cache
import torch
from torch import nn
from nanovllm.models.rope_config import validate_rope_scaling


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        kind = validate_rope_scaling(rope_scaling)
        if kind == "linear":
            inv_freq = inv_freq / rope_scaling["factor"]
        elif kind == "llama3":
            # Llama 3.1/3.2: preserve high frequencies, scale low frequencies,
            # and smoothly interpolate the middle band (HF v4.55.2 reference).
            factor = rope_scaling["factor"]
            low, high = rope_scaling["low_freq_factor"], rope_scaling["high_freq_factor"]
            original = rope_scaling["original_max_position_embeddings"]
            wavelength = 2 * torch.pi / inv_freq
            smooth = (original / wavelength - low) / (high - low)
            middle = (1 - smooth) * inv_freq / factor + smooth * inv_freq
            inv_freq = torch.where(wavelength > original / low, inv_freq / factor,
                                   torch.where(wavelength < original / high, inv_freq, middle))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(head_size, rotary_dim, max_position, base, rope_scaling=None):
    validate_rope_scaling(rope_scaling)
    scaling = tuple(sorted((rope_scaling or {}).items()))
    # Never reuse a CPU reference cache in a later CUDA model construction.
    return _get_rope(head_size, rotary_dim, max_position, base, scaling, str(torch.get_default_device()))


@lru_cache(8)
def _get_rope(head_size, rotary_dim, max_position, base, scaling, device):
    with torch.device(device):
        return RotaryEmbedding(head_size, rotary_dim, max_position, base, dict(scaling) or None)

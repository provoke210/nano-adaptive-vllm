"""Independent mathematical checks plus exact cross-shape kernel regression."""
import os
from types import SimpleNamespace
import pytest


def cuda_torch():
    if not os.environ.get('NANO_MODEL'):
        pytest.skip('explicit CUDA opt-in via NANO_MODEL')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return torch


@pytest.mark.gpu
def test_invariant_linear_matches_reference_and_split_batches():
    torch = cuda_torch()
    from nanovllm.layers.invariant import invariant_linear
    torch.manual_seed(734)
    x = torch.randn(37, 128, device='cuda', dtype=torch.float16)
    weight = torch.randn(257, 128, device='cuda', dtype=torch.float16)
    out = invariant_linear(x, weight, output_dtype=torch.float32)
    ref = x.float() @ weight.float().t()
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    parts = [invariant_linear(part, weight, output_dtype=torch.float32)
             for part in [x[:1], x[1:18], x[18:]]]
    assert torch.equal(out, torch.cat(parts))


@pytest.mark.gpu
def test_attention_paging_chunks_decode_match_full_and_math_reference():
    torch = cuda_torch()
    from nanovllm.layers.invariant import invariant_attention
    torch.manual_seed(893)
    h, kh, d = 4, 2, 128
    lengths = [67, 289]
    total = sum(lengths)
    q = torch.randn(total, h, d, device='cuda', dtype=torch.float16)
    k = torch.randn(total, kh, d, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    tensor = lambda values: torch.tensor(values, device='cuda', dtype=torch.int32)
    ends = tensor([0, 67, total])
    ctx = SimpleNamespace(is_prefill=True, block_tables=None,
                          cu_seqlens_q=ends, cu_seqlens_k=ends, max_seqlen_q=289)
    full = invariant_attention(q, k, v, ctx, kh, d ** -0.5)
    refs = []
    for start, length in [(0, 67), (67, 289)]:
        qq = q[start:start+length].float().transpose(0, 1)
        kk = k[start:start+length].float().repeat_interleave(h//kh, 1).transpose(0, 1)
        vv = v[start:start+length].float().repeat_interleave(h//kh, 1).transpose(0, 1)
        scores = qq @ kk.transpose(1, 2) * d ** -0.5
        mask = torch.ones(length, length, device='cuda', dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, -float('inf'))
        refs.append((scores.softmax(-1) @ vv).transpose(0, 1))
    torch.testing.assert_close(full.float(), torch.cat(refs), rtol=0.002, atol=0.002)
    # Noncontiguous physical pages and an unaligned partial prefill boundary.
    kc = torch.full((5, 256, kh, d), float('nan'), device='cuda', dtype=torch.float16)
    vc = torch.full_like(kc, float('nan'))
    kc[3, :256], kc[1, :33] = k[67:323], k[323:]
    vc[3, :256], vc[1, :33] = v[67:323], v[323:]
    ctx = SimpleNamespace(is_prefill=True, block_tables=tensor([[3, 1]]),
                          cu_seqlens_q=tensor([0, 40]), cu_seqlens_k=tensor([0, 289]), max_seqlen_q=40)
    chunk = invariant_attention(q[-40:], kc, vc, ctx, kh, d ** -0.5)
    assert torch.equal(chunk, full[-40:])
    ctx = SimpleNamespace(is_prefill=False, block_tables=tensor([[3, 1]]), context_lens=tensor([289]))
    decode = invariant_attention(q[-1:], kc, vc, ctx, kh, d ** -0.5)
    assert torch.equal(decode, full[-1:])

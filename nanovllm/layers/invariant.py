"""Fixed reduction layouts for batch-independent single-GPU inference.

No autotuning, split-K, or batch-dependent tile selection. FP32 accumulators;
model activations/KV retain their configured dtype. This experimental path
must be validated against an independent attention reference before use.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M'])
def _linear(X, W, Y, B, M, N: tl.constexpr, K: tl.constexpr,
            XS: tl.constexpr, WS: tl.constexpr, HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    kk = tl.arange(0, 64)
    acc = tl.full((16, 64), 0, tl.float32)
    for start in range(tl.cdiv(K, 64)):
        k = start * 64 + kk
        x = tl.load(X + rows[:, None] * XS + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + cols[None, :] * WS + k[:, None],
                    (cols[None, :] < N) & (k[:, None] < K), 0)
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, 0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


def invariant_linear(x, weight, bias=None, output_dtype=None):
    assert x.ndim == weight.ndim == 2 and x.stride(1) == weight.stride(1) == 1
    m, k = x.shape
    n = weight.shape[0]
    result = torch.empty((m, n), device=x.device, dtype=output_dtype or x.dtype)
    if m:
        _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            x, weight, result, bias if bias is not None else x,
            m, n, k, x.stride(0), weight.stride(0), bias is not None,
            num_warps=4, num_stages=2)
    return result


@triton.jit
def _attention(Q, K, V, O, Q_ENDS, K_ENDS, TABLES,
               QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
               TABLE_STRIDE: tl.constexpr, H: tl.constexpr, KH: tl.constexpr,
               D: tl.constexpr, PAGE: tl.constexpr, SCALE: tl.constexpr,
               PREFILL: tl.constexpr, PAGED: tl.constexpr):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if PREFILL:
        q_start = tl.load(Q_ENDS + seq)
        q_len = tl.load(Q_ENDS + seq + 1) - q_start
        k_len = tl.load(K_ENDS + seq + 1) - tl.load(K_ENDS + seq)
    else:
        q_start, q_len = seq, 1
        k_len = tl.load(K_ENDS + seq)
    if tile * 16 >= q_len:
        return
    qi = tile * 16 + tl.arange(0, 16)
    di = tl.arange(0, D)
    q = tl.load(Q + (q_start + qi[:, None]) * QS + head * D + di[None, :],
                qi[:, None] < q_len, 0)
    key_head = head // (H // KH)
    m = tl.full((16,), -float('inf'), tl.float32)
    normalizer = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, D), 0, tl.float32)
    limit = tl.minimum(k_len, k_len - q_len + (tile + 1) * 16)
    for block in range(tl.cdiv(limit, 64)):
        ki = block * 64 + tl.arange(0, 64)
        if PAGED:
            page = tl.load(TABLES + seq * TABLE_STRIDE + ki // PAGE, ki < k_len, 0)
            slot = page * PAGE + ki % PAGE
        else:
            slot = q_start + ki
        k = tl.load(K + slot[None, :] * KS + key_head * D + di[:, None],
                    ki[None, :] < k_len, 0)
        v = tl.load(V + slot[:, None] * VS + key_head * D + di[None, :],
                    ki[:, None] < k_len, 0)
        scores = tl.dot(q, k) * SCALE
        causal = ki[None, :] <= k_len - q_len + qi[:, None]
        scores = tl.where((ki[None, :] < k_len) & causal, scores, -float('inf'))
        new_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        normalizer = normalizer * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = new_m
    result = acc / tl.maximum(normalizer[:, None], 1e-20)
    tl.store(O + (q_start + qi[:, None]) * H * D + head * D + di[None, :],
             result, qi[:, None] < q_len)


def invariant_attention(q, k, v, context, num_kv_heads, scale):
    """One chronological K reduction order for prefill, mixed, and decode."""
    n, heads, dim = q.shape
    assert dim in (64, 128, 256)
    paged = context.block_tables is not None
    prefill = context.is_prefill
    q_ends = context.cu_seqlens_q if prefill else context.context_lens
    k_ends = context.cu_seqlens_k if prefill else context.context_lens
    seqs = q_ends.numel() - 1 if prefill else n
    max_q = context.max_seqlen_q if prefill else 1
    output = torch.empty_like(q)
    _attention[(seqs, heads, triton.cdiv(max_q, 16))](
        q, k, v, output, q_ends, k_ends,
        context.block_tables if paged else q_ends,
        q.stride(0), num_kv_heads * dim if paged else k.stride(0),
        num_kv_heads * dim if paged else v.stride(0),
        context.block_tables.stride(0) if paged else 0,
        heads, num_kv_heads, dim, k.shape[1] if paged else 256, scale,
        prefill, paged, num_warps=4, num_stages=2)
    return output

import importlib.util
import random
import sys

import pytest

from benchmarks.simulator import SimEngine, CacheOracle
from benchmarks.run import replay
from benchmarks.metrics import summarize
from nanovllm import SamplingParams
from nanovllm.engine.policy import SchedulingOptions, BudgetController


def oracle_output(prompt, count):
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(CacheOracle.next_token(tokens))
    return tokens[len(prompt):]


def check_pool(engine):
    manager = engine.scheduler.block_manager
    seqs = list(engine.scheduler.waiting) + list(engine.scheduler.running)
    refs = [0] * len(manager.blocks)
    for seq in seqs:
        assert 0 <= seq.num_cached_tokens <= seq.num_tokens
        assert len(seq.block_table) == len(set(seq.block_table))
        for block in seq.block_table:
            refs[block] += 1
    assert refs == [block.ref_count for block in manager.blocks]
    assert len(manager.free_block_ids) == len(set(manager.free_block_ids))
    assert set(manager.free_block_ids).isdisjoint(manager.used_block_ids)
    assert set(manager.free_block_ids) | manager.used_block_ids == set(range(len(refs)))
    assert manager.used_block_ids == {i for i, ref in enumerate(refs) if ref}


def drain(engine, limit=20000):
    for _ in range(limit):
        if engine.is_finished():
            check_pool(engine)
            assert not engine.scheduler.block_manager.used_block_ids
            return
        engine.step()
        assert engine.last_step["prefill_tokens"] + engine.last_step["decode_tokens"] <= engine.config.max_num_batched_tokens
        check_pool(engine)
    raise AssertionError("scheduler did not drain")


@pytest.mark.parametrize("mode", ["native", "fixed", "adaptive"])
def test_chunking_prefix_cache_and_generation(mode):
    options = SchedulingOptions(mode=mode, prefill_budget=3, min_prefill_budget=1, max_prefill_budget=16)
    engine = SimEngine(options, blocks=32, block_size=4, token_budget=8, max_seqs=4)
    prompts = [[11] * 12 + [12], [11] * 12 + [13], [17], list(range(13))]
    seqs = [engine.add_request(p, SamplingParams(temperature=0, max_tokens=9)) for p in prompts]
    drain(engine)
    assert [s.completion_token_ids for s in seqs] == [oracle_output(p, 9) for p in prompts]
    # A completed request leaves reusable prefix blocks, not live allocations.
    seq = engine.add_request(prompts[0], SamplingParams(temperature=0, max_tokens=9))
    selected, is_prefill = engine.scheduler.schedule()
    assert seq.num_cached_tokens >= 4
    engine.scheduler.postprocess(selected, engine.runner.run(selected, is_prefill), is_prefill)
    drain(engine)
    assert seq.completion_token_ids == oracle_output(prompts[0], 9)


def test_mixed_decode_and_partial_prefill_never_samples_partial():
    engine = SimEngine(SchedulingOptions(mode="fixed", prefill_budget=2, min_prefill_budget=1),
                       blocks=32, block_size=4, token_budget=8, max_seqs=4)
    old = engine.add_request([3], SamplingParams(max_tokens=8))
    engine.step()
    new = engine.add_request([4] * 10, SamplingParams(max_tokens=3))
    engine.step()
    assert engine.last_step["decode_tokens"] == 1
    assert engine.last_step["prefill_tokens"] == 2
    assert old.num_completion_tokens == 2
    assert new.num_completion_tokens == 0
    drain(engine)


@pytest.mark.parametrize("kv_mode", ["none", "fixed", "adaptive"])
def test_watermark_idle_escape_and_cancellation(kv_mode):
    engine = SimEngine(SchedulingOptions(mode="adaptive", kv_mode=kv_mode, reserve_blocks=100,
                       prefill_budget=2, min_prefill_budget=1), blocks=8, block_size=4)
    one = engine.add_request([1] * 13, SamplingParams(max_tokens=8))
    two = engine.add_request([2] * 9, SamplingParams(max_tokens=8))
    engine.step()
    assert one.num_cached_tokens > 0
    assert engine.scheduler.cancel(one.seq_id)
    assert not engine.scheduler.cancel(one.seq_id)
    drain(engine)
    assert two.completion_token_ids == oracle_output([2] * 9, 8)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("mode", ["native", "fixed", "adaptive"])
def test_random_arrivals_preemption_and_block_recycling(seed, mode):
    rng = random.Random(seed)
    engine = SimEngine(SchedulingOptions(mode=mode, prefill_budget=5, min_prefill_budget=1,
                       max_prefill_budget=16, target_step_ms=0.8,
                       kv_mode="adaptive" if mode == "adaptive" else "none"),
                       blocks=12, block_size=4, token_budget=16, max_seqs=4)
    seqs = []
    for _ in range(20):
        prompt = [rng.randrange(3, 20) for _ in range(rng.randint(1, 20))]
        output = rng.randint(2, 16)
        seqs.append((engine.add_request(prompt, SamplingParams(max_tokens=output)), prompt, output))
        for _ in range(rng.randint(0, 3)):
            if not engine.is_finished():
                engine.step()
                check_pool(engine)
    drain(engine)
    for seq, prompt, output in seqs:
        assert seq.completion_token_ids == oracle_output(prompt, output)
        assert len(seq.token_times) == output


def test_preemption_recomputes_generated_history_without_duplicate_outputs():
    engine = SimEngine(SchedulingOptions(mode="fixed", prefill_budget=16, min_prefill_budget=1),
                       blocks=6, block_size=4, max_seqs=4, token_budget=16)
    seqs = [engine.add_request([i] * 8, SamplingParams(max_tokens=12)) for i in range(3, 6)]
    drain(engine)
    assert sum(s.preemptions for s in seqs) > 0
    assert sum(s.recomputed_tokens for s in seqs) > 0
    assert [s.completion_token_ids for s in seqs] == [oracle_output([i] * 8, 12) for i in range(3, 6)]


def test_adaptive_reserve_uses_block_boundaries():
    engine = SimEngine(SchedulingOptions(mode="fixed", kv_mode="adaptive", decode_lookahead=4),
                       blocks=20, block_size=4)
    engine.add_request([1] * 3, SamplingParams(max_tokens=10))
    engine.step()
    assert engine.scheduler.reserve() == 1


def test_eos_frees_cache():
    engine = SimEngine(SchedulingOptions(mode="fixed"), block_size=4)
    seq = engine.add_request([5], SamplingParams(max_tokens=8))
    engine.scheduler.eos = CacheOracle.next_token([5])
    drain(engine)
    assert len(seq.completion_token_ids) == 1


def test_reject_impossible_request_without_poisoning_queue():
    engine = SimEngine(SchedulingOptions(), blocks=2, block_size=4)
    with pytest.raises(ValueError, match="cannot fit"):
        engine.add_request([1] * 6, SamplingParams(max_tokens=8))
    assert engine.is_finished()


def test_feedback_bounds_recovery_and_aging():
    o = SchedulingOptions(mode="adaptive", prefill_budget=64, min_prefill_budget=16,
                          max_prefill_budget=128, target_step_ms=10, aging_ms=100)
    c = BudgetController(o)
    for _ in range(20):
        c.observe(40, 32)
    assert c.budget == 16
    assert c.choose(100, 101) == 64
    assert c.choose(3, 101) == 3
    for _ in range(100):
        c.observe(1, 64)
    assert c.budget == 128
    c.observe(50, 0)
    assert c.budget == 64
    with pytest.raises(ValueError):
        c.observe(float("nan"), 1)


def test_replay_counts_arrival_delay_and_is_reproducible():
    trace = [dict(id="a", arrival_ms=0.0, token_ids=[4] * 32, max_tokens=3),
             dict(id="b", arrival_ms=0.1, token_ids=[5], max_tokens=2)]
    reports = []
    for _ in range(2):
        e = SimEngine(SchedulingOptions(mode="fixed"))
        records, steps, duration = replay(e, trace, e.clock, e.clock.sleep)
        assert records[1]["token_times_ms"][0] > 0.1
        reports.append((records, steps, duration))
    assert reports[0] == reports[1]


def test_goodput_uses_joint_ttft_tpot_slo():
    rows = [dict(arrival_ms=0, token_times_ms=[5, 25], output_ids=[1, 2], preemptions=0, recomputed_tokens=0),
            dict(arrival_ms=0, token_times_ms=[15], output_ids=[1], preemptions=0, recomputed_tokens=0)]
    m = summarize(rows, 1, ttft_slo_ms=10, tpot_slo_ms=10)
    assert m["goodput_requests_per_s"] == 0
    assert m["output_tokens_per_s"] == 3


def test_cpu_modules_do_not_import_torch():
    # Run in an isolated process because optional GPU tests may import torch.
    import subprocess
    subprocess.run([sys.executable, "-c", "from benchmarks.simulator import SimEngine; import sys; assert 'torch' not in sys.modules"], check=True)

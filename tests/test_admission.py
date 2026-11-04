import pytest
from benchmarks.simulator import SimEngine, CacheOracle
from nanovllm import SamplingParams
from nanovllm.engine.policy import SchedulingOptions


def expected(seq):
    tokens = list(seq.prompt_token_ids)
    for _ in range(seq.max_tokens):
        tokens.append(CacheOracle.next_token(tokens))
    return tokens[-seq.max_tokens:]


def test_aged_prefill_enters_full_decode_batch_and_decodes_rotate():
    engine = SimEngine(SchedulingOptions(mode='fixed', prefill_budget=32, admission_wait_ms=1),
                       max_seqs=2, blocks=64, block_size=4, token_budget=32)
    old = [engine.add_request([i+11]*4, SamplingParams(max_tokens=30)) for i in range(2)]
    engine.step()
    new = engine.add_request([31]*4, SamplingParams(max_tokens=10))
    engine.clock.sleep(.002)
    engine.step()
    assert new.num_completion_tokens == 1
    assert all(s.num_completion_tokens < 30 for s in old)
    assert engine.last_step['prefill_slot_reserved']
    assert len(engine.scheduler.running) == 3
    before = [s.num_completion_tokens for s in old + [new]]
    engine.step()
    engine.step()
    assert all(s.num_completion_tokens > n for s, n in zip(old + [new], before))
    while not engine.is_finished():
        engine.step()
        assert engine.last_step['decode_tokens'] <= 2
    assert all(s.completion_token_ids == expected(s) for s in old + [new])


@pytest.mark.parametrize('max_seqs,blocks', [(1, 16), (3, 12), (3, 64)])
def test_admission_rotation_preserves_cache_under_pressure(max_seqs, blocks):
    engine = SimEngine(SchedulingOptions(mode='adaptive', controller='cost', prefill_budget=32,
                       admission_wait_ms=1, kv_mode='adaptive', decode_lookahead=4),
                       max_seqs=max_seqs, blocks=blocks, block_size=4, token_budget=32)
    requests = [engine.add_request([i+10]*8, SamplingParams(max_tokens=20)) for i in range(8)]
    for _ in range(5000):
        if engine.is_finished():
            break
        engine.step()
        assert engine.last_step['decode_tokens'] <= max_seqs
        assert engine.last_step['prefill_tokens'] + engine.last_step['decode_tokens'] <= 32
    assert engine.is_finished()
    assert all(s.completion_token_ids == expected(s) for s in requests)
    assert not engine.scheduler.block_manager.used_block_ids


def test_completion_forecast_avoids_unneeded_reservation_then_batches_waiters():
    engine = SimEngine(SchedulingOptions(mode='fixed', prefill_budget=32, admission_wait_ms=1,
                       admission_target_ms=100, admission_max_slots=3),
                       max_seqs=4, blocks=128, block_size=4, token_budget=32)
    old = [engine.add_request([i+11]*4, SamplingParams(max_tokens=30)) for i in range(4)]
    engine.step()
    new = [engine.add_request([i+21]*4, SamplingParams(max_tokens=10)) for i in range(4)]
    engine.clock.sleep(.002)
    engine.scheduler.decode_step_ms = 1  # completion soon, queue can wait
    engine.step()
    assert not any(s.num_completion_tokens for s in new)
    assert not engine.last_step['prefill_slot_reserved']
    engine.scheduler.decode_step_ms = 10  # same requests on a slower path
    engine.step()
    assert engine.last_step['prefill_slots_reserved'] == 3
    assert sum(bool(s.num_completion_tokens) for s in new) == 3
    assert engine.last_step['decode_tokens'] == 1
    for _ in range(300):
        if engine.is_finished():
            break
        engine.step()
    assert engine.is_finished()
    assert all(s.completion_token_ids == expected(s) for s in old + new)

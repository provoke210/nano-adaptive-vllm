import pytest
from nanovllm.engine.cost_model import StepCostModel


def test_separates_prefill_overhead_from_token_work():
    model = StepCostModel()
    for tokens in [64, 256, 512, 1024, 2048]:
        model.observe(28 + 0.006 * tokens, tokens)
    assert model.intercept_ms == pytest.approx(28)
    assert model.token_ms == pytest.approx(0.006)
    for _ in range(10):
        model.observe(4, 0)
    assert model.decode_ms == 4
    assert model.predict(1000) == pytest.approx(34)
    budget, reason = model.affordable_budget(20, 32, 2048)
    assert (budget, reason) == (2048, 'target_below_overhead')
    budget, reason = model.affordable_budget(34, 32, 2048)
    assert 999 <= budget <= 1000


def test_identical_shapes_cannot_determine_intercept():
    model = StepCostModel()
    for _ in range(10):
        model.observe(30, 512)
    assert model.affordable_budget(40, 32, 2048) == (None, 'uncalibrated')


def test_recent_window_adapts_to_cost_change():
    model = StepCostModel(window=8)
    for floor, slope in [(28, 0.006), (10, 0.02)]:
        for tokens in [64, 128, 256, 512, 1024, 2048, 512, 128]:
            model.observe(floor + slope * tokens, tokens)
    assert model.predict(1000) == pytest.approx(30)
    assert model.intercept_ms == pytest.approx(10)


@pytest.mark.parametrize('duration,tokens', [(0, 0), (float('nan'), 1), (1, -1)])
def test_rejects_invalid_observations(duration, tokens):
    with pytest.raises(ValueError):
        StepCostModel().observe(duration, tokens)


def test_cost_controller_coalescing_never_idles_or_starves():
    from nanovllm.engine.policy import SchedulingOptions, CostBudgetController
    c = CostBudgetController(SchedulingOptions(mode='adaptive', controller='cost', coalesce_ms=24))
    for tokens in [64, 128, 512, 1024]:
        c.observe(28 + 0.006*tokens, tokens)
    c.observe(4, 0)
    assert c.budget == 2048  # 20ms cannot cover 28ms launch overhead.
    assert c.defer(100, 10, 3, False)
    assert not c.defer(100, 24, 3, False)
    assert not c.defer(100, 0, 0, False)
    assert not c.defer(100, 0, 3, True)
    assert not c.defer(700, 0, 3, False)
    assert c.choose(17, 0) == 17
    assert c.reason == 'target_below_overhead'


def test_cost_scheduler_completes_with_identical_oracle_outputs():
    from nanovllm.engine.policy import SchedulingOptions
    from nanovllm import SamplingParams
    from benchmarks.simulator import SimEngine, CacheOracle
    engine = SimEngine(SchedulingOptions(mode='adaptive', controller='cost', coalesce_ms=24),
                       blocks=16, token_budget=64, max_seqs=4, block_size=4)
    sequences = []
    for i in range(20):
        if i % 4 == 0:
            sequences.append(engine.add_request([i+2]*8, SamplingParams(max_tokens=12)))
        if not engine.is_finished():
            engine.step()
    for _ in range(500):
        if engine.is_finished():
            break
        engine.step()
    assert engine.is_finished()
    for seq in sequences:
        expected = list(seq.prompt_token_ids)
        for _ in range(12):
            expected.append(CacheOracle.next_token(expected))
        assert seq.completion_token_ids == expected[-12:]

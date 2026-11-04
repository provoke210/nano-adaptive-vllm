from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
from nanovllm.engine.input_buffers import DecodeInputBuffers
from nanovllm.engine.sequence import Sequence
from nanovllm import SamplingParams
from nanovllm.engine.profiling import StageTrace, StageProfiler


def seq(tokens, table, temperature=0):
    result = Sequence(tokens, SamplingParams(temperature=temperature))
    result.block_table = table
    return result


def test_decode_inputs_ragged_resize_and_pointer_stability():
    pool = DecodeInputBuffers(4, 20, 4, device='cpu')
    pointers = {k: (v.data_ptr(), pool.gpu[k].data_ptr()) for k, v in pool.host.items()}
    sequences = [seq([1, 2, 3, 4, 5], [7, 2]), seq([8], [4], .8)]
    n, width = pool.fill(sequences)
    out = pool.upload(n, width)
    assert out['input_ids'].tolist() == [5, 8]
    assert out['positions'].tolist() == [4, 0]
    assert out['slot_mapping'].tolist() == [8, 16]
    assert out['context_lens'].tolist() == [5, 1]
    assert out['block_tables'].tolist() == [[7, 2], [4, -1]]
    assert out['temperatures'].tolist() == pytest.approx([0, .8])
    n, width = pool.fill([seq([9, 10, 11, 12], [3])])
    out = pool.upload(n, width)
    assert out['block_tables'].shape == (1, 1)
    assert out['slot_mapping'].tolist() == [15]
    for key, value in pool.host.items():
        assert (value.data_ptr(), pool.gpu[key].data_ptr()) == pointers[key]


@pytest.mark.parametrize("inference", [False, True])
def test_graph_padding_and_direct_copy(inference):
    with torch.inference_mode(inference):
        pool = DecodeInputBuffers(4, 20, 4, device='cpu')
        graph = pool.gpu
        for value in graph.values():
            value.fill_(99)
    n, width = pool.fill([seq([3], [2]), seq([4]*9, [5, 6, 7])])
    out = pool.upload(n, width)
    assert graph['slot_mapping'].tolist() == [8, 28, -1, -1]
    assert graph['context_lens'].tolist() == [1, 9, 0, 0]
    assert out['input_ids'].data_ptr() == graph['input_ids'].data_ptr()
    assert out['block_tables'].tolist() == [[2, -1, -1], [5, 6, 7]]


def test_inflight_host_guard_waits_before_refill():
    pool = DecodeInputBuffers(2, 8, 4, device='cpu')
    calls = []
    pool.pending_stream = SimpleNamespace(synchronize=lambda: calls.append('wait'))
    pool.fill([seq([1], [0])])
    assert calls == ['wait']
    pool.pending_stream = SimpleNamespace(synchronize=lambda: calls.append('unexpected'))
    pool.mark_consumed()
    pool.fill([seq([2], [1])])
    assert calls == ['wait']


@pytest.mark.parametrize('sequences', [[], [seq([1], [0])]*3, [seq([1]*9, [0, 1, 2])], [seq([1]*5, [0])]])
def test_reject_bad_capacities(sequences):
    pool = DecodeInputBuffers(2, 8, 4, device='cpu')
    with pytest.raises(ValueError):
        pool.fill(sequences)


def test_deferred_profile_events_do_not_add_stage_syncs():
    calls, events = [], []
    class FakeEvent:
        done = False
        def __init__(self):
            self.index = len(events)
            events.append(self)
        def record(self):
            calls.append('record')
        def query(self):
            return self.done
        def synchronize(self):
            calls.append('sync')
            for event in events:
                event.done = True
        def elapsed_time(self, other):
            assert other.done
            return float(other.index-self.index)
    trace = StageTrace(FakeEvent, step=1)
    trace.mark('prepare'); trace.mark('model'); trace.mark('sample'); trace.mark('result_wait')
    profiler = StageProfiler()
    profiler.submit(trace)
    profiler.collect()
    assert calls.count('sync') == 0 and not profiler.ready
    records = profiler.collect(wait=True)
    assert calls.count('sync') == 1
    assert records[0]['cuda_stream_ms'] == dict(prepare=1., model=1., sample=1., result_wait=1.)
    assert all(value >= 0 for value in records[0]['cpu_ms'].values())
    assert not profiler.pending and not profiler.ready


def test_runner_uses_direct_graph_buffers_and_temperatures():
    pytest.importorskip('transformers')
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import get_context, reset_context
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(reuse_decode_buffers=True)
    runner.enforce_eager = False
    runner.decode_buffers = DecodeInputBuffers(4, 20, 4, device='cpu')
    runner.graph_vars = runner.decode_buffers.gpu
    runner._decode_temperatures = None
    sequences = [seq([1, 2, 3, 4, 5], [3, 1])]
    try:
        ids, positions = runner.prepare_decode(sequences)
        assert runner._direct_decode_graph
        assert ids.data_ptr() == runner.graph_vars['input_ids'].data_ptr()
        assert positions.tolist() == [4]
        assert get_context().slot_mapping.tolist() == [4]
        assert runner.prepare_sample(sequences).tolist() == [0]
    finally:
        reset_context()


def test_slab_views_aligned_contiguous_disjoint_and_single_copy():
    pool = DecodeInputBuffers(3, 19, 4, device='cpu')
    ranges = []
    for name, tensor in pool.host.items():
        offset = tensor.data_ptr() - pool.host_storage.data_ptr()
        assert offset % 8 == 0 and tensor.is_contiguous()
        assert tensor.dtype == pool.gpu[name].dtype
        ranges.append((offset, offset+tensor.numel()*tensor.element_size()))
    ranges.sort()
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
    n, width = pool.fill([seq([3]*5, [2, 3], .7)])
    class CopySpy:
        calls = 0
        def copy_(self, source, non_blocking):
            self.calls += 1
            assert source is pool.host_storage and non_blocking
    spy = CopySpy()
    pool.gpu_storage = spy
    pool.upload(n, width)
    assert spy.calls == 1

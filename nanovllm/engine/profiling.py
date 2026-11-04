"""Opt-in wall-clock phases and deferred CUDA event spans; never add stage barriers."""
from time import perf_counter


class StageTrace:
    def __init__(self, event_factory, **metadata):
        self.event_factory = event_factory
        self.record = metadata
        self.events = []
        self.names = []
        self.cpu = [perf_counter()]
        self._event()

    def _event(self):
        event = self.event_factory()
        event.record()
        self.events.append(event)

    def mark(self, name):
        self.cpu.append(perf_counter())
        self.names.append(name)
        self._event()

    def resolve(self):
        # CUDA spans include launch starvation/stream waits, not just kernel execution.
        self.record["cpu_ms"] = {name: (self.cpu[i+1]-self.cpu[i])*1000 for i, name in enumerate(self.names)}
        self.record["cuda_stream_ms"] = {name: self.events[i].elapsed_time(self.events[i+1]) for i, name in enumerate(self.names)}
        return self.record


class StageProfiler:
    def __init__(self):
        self.pending = []
        self.ready = []
        self.sequence = 0

    def begin(self, **metadata):
        import torch
        self.sequence += 1
        return StageTrace(lambda: torch.cuda.Event(enable_timing=True), step=self.sequence, **metadata)

    def submit(self, trace):
        self.pending.append(trace)
        # Keep CUDA event memory bounded in long profiling runs; no forced barrier.
        if len(self.pending) >= 128:
            self.collect()

    def collect(self, wait=False):
        if wait and self.pending:
            self.pending[-1].events[-1].synchronize()
        while self.pending and self.pending[0].events[-1].query():
            self.ready.append(self.pending.pop(0).resolve())
        if wait:
            result, self.ready = self.ready, []
            return result

"""Synthetic timing + token-cache oracle, NOT a GPU performance predictor."""
from types import SimpleNamespace
from nanovllm.engine.adaptive_scheduler import make_scheduler
from nanovllm.engine.sequence import Sequence


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class CacheOracle:
    def __init__(self, config, clock):
        self.clock = clock
        self.block_size = config.kvcache_block_size
        self.cache = [[None] * self.block_size for _ in range(config.num_kvcache_blocks)]

    @staticmethod
    def next_token(tokens):
        return (sum(tokens) + 17 * len(tokens)) % 997 + 3

    def run(self, seqs, is_prefill):
        result = []
        prefills = decodes = attention_work = 0
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            assert 0 <= start < end <= seq.num_tokens
            assert end <= len(seq.block_table) * self.block_size
            for i in range(start):
                assert self.cache[seq.block_table[i // self.block_size]][i % self.block_size] == seq[i], "stale/uncomputed KV prefix"
            for i in range(start, end):
                self.cache[seq.block_table[i // self.block_size]][i % self.block_size] = seq[i]
            if seq.is_prefill:
                prefills += end - start
            else:
                assert end - start == 1
                decodes += 1
            attention_work += (end - start) * (start + end) / 2
            result.append(self.next_token(seq.token_ids) if end == seq.num_tokens else None)
        # Fixed, policy-independent illustrative costs; do not calibrate these
        # to manufacture a winner. CUDA Graph/kernel effects are not modeled.
        duration_ms = 0.4 + 0.015 * prefills + 0.06 * decodes + 0.000002 * attention_work
        self.clock.sleep(duration_ms / 1000)
        return result


class SimEngine:
    def __init__(self, options, blocks=64, block_size=256, token_budget=2048,
                 max_seqs=16, max_model_len=4096):
        self.clock = VirtualClock()
        Sequence.block_size = block_size
        self.config = SimpleNamespace(scheduling=options, num_kvcache_blocks=blocks,
                                      kvcache_block_size=block_size, max_num_seqs=max_seqs,
                                      max_num_batched_tokens=token_budget, max_model_len=max_model_len,
                                      eos=-1)
        self.scheduler = make_scheduler(self.config, clock=self.clock)
        self.runner = CacheOracle(self.config, self.clock)
        self.last_step = {}

    def add_request(self, tokens, params):
        seq = Sequence(tokens, params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        start = self.clock()
        seqs, is_prefill = self.scheduler.schedule()
        prefills = sum(s.num_scheduled_tokens for s in seqs if s.is_prefill)
        decodes = sum(s.num_scheduled_tokens for s in seqs if not s.is_prefill)
        result = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, result, is_prefill)
        elapsed = (self.clock() - start) * 1000
        self.scheduler.observe(elapsed, prefills)
        self.last_step = dict(elapsed_ms=elapsed, prefill_tokens=prefills, decode_tokens=decodes,
                              used_blocks=len(self.scheduler.block_manager.used_block_ids),
                              **getattr(self.scheduler, "last_decision", {}))

    def is_finished(self):
        return self.scheduler.is_finished()

    def exit(self):
        pass

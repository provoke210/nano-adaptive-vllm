from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: "Config", clock=perf_counter):
        self.clock = clock
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, value):
        self._eos = value
        self.eos_token_ids = frozenset(value if isinstance(value, (list, tuple)) else [value])

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.block_size != self.block_size:
            raise ValueError("Sequence and cache block sizes must match")
        if seq.num_tokens + seq.max_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_tokens exceeds max_model_len")
        needed = (seq.num_tokens + seq.max_tokens - 2) // self.block_size + 1
        if needed > len(self.block_manager.blocks):
            raise ValueError("request cannot fit KV pool even when running alone")
        seq.arrival_time = self.clock()
        self.waiting.append(seq)

    def cancel(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    self.block_manager.deallocate(seq)
                    seq.status = SequenceStatus.FINISHED
                    return True
        return False

    def observe(self, elapsed_ms: float, prefill_tokens: int):
        pass

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.preemptions += 1
        seq.recompute_until = max(seq.recompute_until, seq.num_cached_tokens)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if len(seqs) != len(token_ids):
            raise ValueError("runner returned the wrong number of results")
        for seq, token_id in zip(seqs, token_ids):
            seq.recomputed_tokens += max(0, min(seq.num_cached_tokens + seq.num_scheduled_tokens,
                                               seq.recompute_until) - seq.num_cached_tokens)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            if token_id is None:
                raise ValueError("missing sampled token for a completed chunk")
            seq.append_token(token_id)
            seq.token_times.append(self.clock())
            if (not seq.ignore_eos and token_id in self.eos_token_ids) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

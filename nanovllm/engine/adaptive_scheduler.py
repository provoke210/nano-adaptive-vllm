"""Decode-first mixed batching using upstream paged varlen attention.

Full input block allocation and prefix hashing are inherited from nano-vLLM.
Pure decode keeps the CUDA Graph path; mixed batches use eager varlen.
"""
from math import ceil
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.engine.policy import BudgetController, CostBudgetController


class AdaptiveScheduler(Scheduler):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.options = config.scheduling
        controller = CostBudgetController if self.options.controller == 'cost' else BudgetController
        self.controller = controller(self.options)
        self.last_decision = {}
        self.decode_step_ms = None

    def observe(self, elapsed_ms, prefill_tokens):
        self.controller.observe(elapsed_ms, prefill_tokens)
        if not prefill_tokens:
            self.decode_step_ms = elapsed_ms if self.decode_step_ms is None else (
                0.2 * elapsed_ms + 0.8 * self.decode_step_ms)

    def reserve(self):
        o = self.options
        if o.kv_mode == "none":
            return 0
        if o.kv_mode == "fixed":
            return o.reserve_blocks
        # Reserve future growth of requests already holding KV. Prefix blocks
        # are counted by physical allocation in BlockManager.can_allocate.
        growth = 0
        for seq in list(self.running) + list(self.waiting):
            if not seq.block_table:
                continue
            remaining = max(0, seq.max_tokens - seq.num_completion_tokens - 1)
            future = seq.num_tokens + min(o.decode_lookahead, remaining)
            growth += max(0, ceil(future / self.block_size) - len(seq.block_table))
        return o.reserve_blocks + growth

    def schedule(self):
        selected = []
        decode_count = 0
        age_ms = max(0, (self.clock() - self.waiting[0].arrival_time) * 1000) if self.waiting else 0
        # A full Decode batch must not prevent every new request from getting
        # its first token until an old generation finishes. Reserve one batch
        # slot once the queue ages, only when the head can plausibly enter KV.
        release_ms = None
        if self.running and self.decode_step_ms is not None:
            # max_tokens is a client cap, not an oracle for actual EOS length.
            # This upper-bound-based estimate can reserve slots too early.
            remaining = min(s.max_tokens - s.num_completion_tokens for s in self.running)
            release_ms = remaining * self.decode_step_ms * max(1, len(self.running) / self.max_num_seqs)
        urgent = (not self.options.admission_target_ms or age_ms >= self.options.admission_target_ms
                  or release_ms is not None and age_ms + release_ms >= self.options.admission_target_ms)
        reserve_slot = bool(self.options.admission_wait_ms and self.waiting and urgent
            and len(self.running) >= self.max_num_seqs
            and self.max_num_seqs > 1 and age_ms >= self.options.admission_wait_ms
            and (self.waiting[0].block_table or
                 self.block_manager.can_allocate(self.waiting[0], self.reserve()) >= 0))
        reserved_slots = min(len(self.waiting), self.options.admission_max_slots,
                             self.max_num_seqs - 1) if reserve_slot else 0
        decode_quota = self.max_num_seqs - reserved_slots
        # Reclaim incomplete prefills first, then unselected running tails.
        while self.running and len(selected) < decode_quota:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                partials = [s for s in self.waiting if s.block_table]
                if partials:
                    victim = partials[-1]
                    self.waiting.remove(victim)
                    self.preempt(victim)
                elif self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                selected.append(seq)
                decode_count += 1
        if self.options.admission_wait_ms:
            # The resident set may now exceed the per-step sequence cap.
            # Rotate selected requests behind those skipped in this step.
            self.running.extend(selected)
        else:
            self.running.extendleft(reversed(selected))

        age_ms = max(0, (self.clock() - self.waiting[0].arrival_time) * 1000) if self.waiting else 0
        budget = self.controller.choose(self.max_num_batched_tokens - decode_count, age_ms)
        requested_budget = budget
        if isinstance(self.controller, CostBudgetController) and self.waiting:
            partial = any(s.block_table for s in self.waiting)
            queued_tokens = sum(s.num_tokens - s.num_cached_tokens for s in self.waiting)
            if self.controller.defer(queued_tokens, age_ms, decode_count, partial):
                budget = requested_budget = 0
        reserve = self.reserve()
        # If completely idle, let the head request start regardless of the
        # soft watermark. Hard allocation limits are never bypassed.
        if not self.running and not any(s.block_table for s in self.waiting):
            reserve = 0
        for seq in list(self.waiting):
            if budget == 0 or len(selected) == self.max_num_seqs:
                break
            if not seq.block_table:
                cached = self.block_manager.can_allocate(seq, reserve)
                if cached == -1:
                    break  # FCFS, no silent bypass/starvation of long prompts
                self.block_manager.allocate(seq, cached)
            count = min(seq.num_tokens - seq.num_cached_tokens, budget)
            seq.is_prefill = True
            seq.num_scheduled_tokens = count
            budget -= count
            selected.append(seq)
            if seq.num_cached_tokens + count == seq.num_tokens:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
        if not selected:
            raise RuntimeError("scheduler made no progress; check capacity and budgets")
        self.last_decision = {
            "prefill_budget": requested_budget, "reserve_blocks": reserve,
            "decode_sequences": decode_count,
            "prefill_slot_reserved": reserve_slot,
            "prefill_slots_reserved": reserved_slots,
            "estimated_slot_release_ms": release_ms,
            "reason": self.controller.reason if self.options.mode == "adaptive" else "fixed",
        }
        if isinstance(self.controller, CostBudgetController):
            model = self.controller.model
            self.last_decision.update(cost_intercept_ms=model.intercept_ms,
                cost_per_token_ms=model.token_ms, cost_error_ms=model.error_ms)
        return selected, any(seq.is_prefill for seq in selected)


def make_scheduler(config, **kwargs):
    cls = Scheduler if config.scheduling.mode == "native" else AdaptiveScheduler
    return cls(config, **kwargs)

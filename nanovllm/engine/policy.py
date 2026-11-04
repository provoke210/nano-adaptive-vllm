"""CPU-only feedback controller. Heuristic, not a latency/SLO guarantee."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SchedulingOptions:
    mode: str = "native"
    prefill_budget: int = 512
    min_prefill_budget: int = 32
    max_prefill_budget: int = 2048
    target_step_ms: float = 20.0
    recovery_steps: int = 3
    aging_ms: float = 500.0
    kv_mode: str = "none"
    reserve_blocks: int = 0
    decode_lookahead: int = 16
    controller: str = "aimd"
    coalesce_ms: float = 0.0
    coalesce_tokens: int = 512
    admission_wait_ms: float = 0.0
    admission_target_ms: float = 0.0
    admission_max_slots: int = 1

    def __post_init__(self):
        if self.mode not in {"native", "fixed", "adaptive"}:
            raise ValueError("mode must be native, fixed or adaptive")
        if self.kv_mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("kv_mode must be none, fixed or adaptive")
        if self.controller not in {"aimd", "cost"}:
            raise ValueError("controller must be aimd or cost")
        if not math.isfinite(self.coalesce_ms) or self.coalesce_ms < 0:
            raise ValueError("coalesce_ms must be finite and nonnegative")
        if not math.isfinite(self.admission_wait_ms) or self.admission_wait_ms < 0:
            raise ValueError("admission_wait_ms must be finite and nonnegative")
        if not math.isfinite(self.admission_target_ms) or self.admission_target_ms < 0:
            raise ValueError("admission_target_ms must be finite and nonnegative")
        if type(self.admission_max_slots) is not int or self.admission_max_slots < 1:
            raise ValueError("admission_max_slots must be a positive integer")
        if type(self.coalesce_tokens) is not int or self.coalesce_tokens < 1:
            raise ValueError("coalesce_tokens must be a positive integer")
        for name in ("prefill_budget", "min_prefill_budget", "max_prefill_budget",
                     "recovery_steps", "decode_lookahead"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.min_prefill_budget <= self.prefill_budget <= self.max_prefill_budget:
            raise ValueError("Require min <= initial/fixed budget <= max")
        for value in (self.target_step_ms, self.aging_ms):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("latency thresholds must be finite and positive")
        if self.reserve_blocks < 0:
            raise ValueError("reserve_blocks must be nonnegative")
        if self.mode == "native" and self.kv_mode != "none":
            raise ValueError("native baseline does not apply KV admission policies")


class BudgetController:
    def __init__(self, options: SchedulingOptions):
        self.options = options
        self.budget = options.prefill_budget
        self.ewma_ms = None
        self.low_steps = 0
        self.reason = "initial"

    def observe(self, elapsed_ms: float, prefill_tokens: int):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError("step duration must be finite and positive")
        o = self.options
        if o.mode != "adaptive":
            return
        # Only mixed/prefill samples can justify growth. Slow decode-only
        # steps can still reduce the budget when new requests arrive.
        if prefill_tokens:
            self.ewma_ms = elapsed_ms if self.ewma_ms is None else 0.3 * elapsed_ms + 0.7 * self.ewma_ms
        pressure_ms = max(elapsed_ms, self.ewma_ms or 0)
        if pressure_ms > o.target_step_ms:
            self.budget = max(o.min_prefill_budget, self.budget // 2)
            self.low_steps = 0
            self.reason = "latency_backoff"
        elif prefill_tokens and pressure_ms < 0.8 * o.target_step_ms:
            self.low_steps += 1
            if self.low_steps >= o.recovery_steps:
                self.budget = min(o.max_prefill_budget, self.budget + o.min_prefill_budget)
                self.low_steps = 0
                self.reason = "latency_recovery"
        else:
            self.low_steps = 0

    def choose(self, remaining: int, oldest_wait_ms: float) -> int:
        budget = self.budget
        if self.options.mode == "adaptive" and oldest_wait_ms >= self.options.aging_ms:
            # Give an old head request at least the initial budget, subject to
            # the hard token cap. This intentionally trades ITL for progress.
            budget = max(budget, self.options.prefill_budget)
            self.reason = "aged_request_floor"
        return min(max(remaining, 0), budget)


class CostBudgetController:
    """Learn marginal token cost separately from the fixed prefill overhead."""
    def __init__(self, options):
        from nanovllm.engine.cost_model import StepCostModel
        self.options = options
        self.model = StepCostModel()
        self.budget = options.prefill_budget
        self.reason = 'cost_warmup'
        self.budget_reason = self.reason

    def observe(self, elapsed_ms, prefill_tokens):
        self.model.observe(elapsed_ms, prefill_tokens)
        if self.options.mode != 'adaptive':
            return
        estimate, reason = self.model.affordable_budget(
            self.options.target_step_ms, self.options.min_prefill_budget,
            self.options.max_prefill_budget)
        if estimate is not None:
            self.budget = estimate
            self.reason = reason
            self.budget_reason = reason
        elif prefill_tokens:
            # Probe larger chunks instead of interpreting the fixed overhead
            # of an eager prefill as evidence that every chunk is too large.
            self.budget = min(self.options.max_prefill_budget, self.budget * 2)

    def choose(self, remaining, oldest_wait_ms):
        self.reason = self.budget_reason
        budget = self.budget
        if oldest_wait_ms >= self.options.aging_ms:
            budget = max(budget, self.options.prefill_budget)
            self.reason = 'aged_request_floor'
        return min(max(remaining, 0), budget)

    def defer(self, queued_tokens, oldest_wait_ms, decode_count, partial):
        # Never idle the GPU or interrupt a partial prefill just to form a batch.
        if not decode_count or partial or not self.options.coalesce_ms:
            return False
        delay = min(self.options.coalesce_ms, self.options.aging_ms)
        efficient = min(self.budget, self.options.coalesce_tokens)
        if queued_tokens < efficient and oldest_wait_ms < delay:
            self.reason = 'coalesce_while_decoding'
            return True
        return False

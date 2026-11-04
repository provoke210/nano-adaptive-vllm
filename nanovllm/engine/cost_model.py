"""Online step-cost model; a latency estimate, never an SLO guarantee."""
from collections import deque
import math


class StepCostModel:
    """Fit elapsed = launch/dispatch overhead + cost_per_token * prefill.

    Small prefills retain the launch overhead. Decode-only observations have a
    different execution path and must not pull down the prefill cost estimate.
    Samples are bounded to adapt when the active workload changes.
    """
    def __init__(self, window=32):
        self.samples = deque(maxlen=window)
        self.decode_ms = None
        self.intercept_ms = None
        self.token_ms = None
        self.error_ms = 0.0

    def observe(self, elapsed_ms, prefill_tokens):
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0 or prefill_tokens < 0:
            raise ValueError('positive finite duration and nonnegative tokens required')
        if prefill_tokens == 0:
            self.decode_ms = elapsed_ms if self.decode_ms is None else 0.2 * elapsed_ms + 0.8 * self.decode_ms
            return
        self.samples.append((prefill_tokens, elapsed_ms))
        if len(self.samples) < 4:
            return
        mean_x = sum(x for x, _ in self.samples) / len(self.samples)
        mean_y = sum(y for _, y in self.samples) / len(self.samples)
        variance = sum((x - mean_x) ** 2 for x, _ in self.samples)
        # Identical-size observations cannot identify fixed and marginal cost.
        if variance < len(self.samples) * 32 ** 2:
            return
        slope = sum((x - mean_x) * (y - mean_y) for x, y in self.samples) / variance
        self.token_ms = max(0.0, slope)
        self.intercept_ms = max(0.0, mean_y - self.token_ms * mean_x)
        residuals = sorted(abs(y - self.predict(x)) for x, y in self.samples)
        self.error_ms = residuals[len(residuals) // 2]

    def predict(self, tokens):
        if self.intercept_ms is None:
            return None
        return self.intercept_ms + self.token_ms * tokens

    def affordable_budget(self, target_ms, minimum, maximum):
        if self.intercept_ms is None:
            return None, 'uncalibrated'
        available = target_ms - self.intercept_ms - 2 * self.error_ms
        if available <= 0:
            # A smaller chunk cannot meet the target. Amortize fixed overhead
            # and surface infeasibility rather than silently collapsing to 32.
            return maximum, 'target_below_overhead'
        if self.token_ms <= 1e-9:
            return maximum, 'overhead_dominated'
        raw = int(available / self.token_ms)
        return max(minimum, min(maximum, raw)), 'cost_estimate'

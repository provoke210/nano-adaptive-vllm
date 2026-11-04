import math


def percentile(values, q):
    if not values:
        return None
    items = sorted(values)
    index = (len(items) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return items[lo] + (items[hi] - items[lo]) * (index - lo)


def summarize(records, duration_s, ttft_slo_ms, tpot_slo_ms):
    completed = [r for r in records if r["token_times_ms"]]
    ttfts, tpots, itls, e2es = [], [], [], []
    good = 0
    for r in completed:
        times = r["token_times_ms"]
        ttft = times[0] - r["arrival_ms"]
        intervals = [b - a for a, b in zip(times, times[1:])]
        tpot = sum(intervals) / len(intervals) if intervals else 0.0
        ttfts.append(ttft)
        tpots.append(tpot)
        itls.extend(intervals)
        e2es.append(times[-1] - r["arrival_ms"])
        good += ttft <= ttft_slo_ms and tpot <= tpot_slo_ms
    return dict(requests=len(records), completed=len(completed), duration_s=duration_s,
                output_tokens_per_s=sum(len(r["output_ids"]) for r in records) / duration_s,
                goodput_requests_per_s=good / duration_s, slo_attainment=good / len(records),
                ttft_p50_ms=percentile(ttfts, 0.5), ttft_p99_ms=percentile(ttfts, 0.99),
                tpot_p99_ms=percentile(tpots, 0.99), itl_p99_ms=percentile(itls, 0.99),
                e2e_p99_ms=percentile(e2es, 0.99),
                preemptions=sum(r["preemptions"] for r in records),
                recomputed_tokens=sum(r["recomputed_tokens"] for r in records))

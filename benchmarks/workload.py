import argparse
import json
import math
from pathlib import Path
import random


def generate(count=60, seed=7, scenario="mixed", rate=15.0):
    if count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("count and rate must be positive")
    rng = random.Random(seed)
    arrival = 0.0
    rows = []
    for i in range(count):
        if i:
            arrival += rng.expovariate(rate) * 1000
        long = scenario != "short" and rng.random() < 0.3
        length = rng.randint(768, 1536) if long else rng.randint(32, 128)
        output = rng.randint(64, 128) if scenario != "pressure" else rng.randint(256, 512)
        if scenario == "burst":
            at = (i // 12) * 1000.0
        else:
            at = arrival
        rows.append(dict(id=str(i), arrival_ms=round(at, 4),
                         token_ids=[rng.randrange(10, 1000) for _ in range(length)],
                         max_tokens=output))
    return rows


def load_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("trace cannot be empty")
    ids = set()
    previous = -1.0
    for row in rows:
        if not isinstance(row["id"], str) or row["id"] in ids:
            raise ValueError("trace IDs must be unique strings")
        ids.add(row["id"])
        at = row["arrival_ms"]
        if not math.isfinite(at) or at < 0 or at < previous:
            raise ValueError("arrival_ms must be finite, nonnegative, and sorted")
        previous = at
        if not row["token_ids"] or any(type(x) is not int or x < 0 for x in row["token_ids"]):
            raise ValueError("token_ids must be nonempty nonnegative integers")
        if type(row["max_tokens"]) is not int or row["max_tokens"] < 1:
            raise ValueError("max_tokens must be positive")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--rate", type=float, default=15)
    p.add_argument("--scenario", choices=["short", "mixed", "burst", "pressure"], default="mixed")
    a = p.parse_args()
    rows = generate(a.count, a.seed, a.scenario, a.rate)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} synthetic token requests to {dest}")


if __name__ == "__main__":
    main()

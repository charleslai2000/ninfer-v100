#!/usr/bin/env python3
"""S4 benchmark for frozen S1 V100. Writes compact, validated JSONL per wave."""
import concurrent.futures as cf
import json
import os
import statistics
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

BASE = os.environ.get("S4_BASE", "http://127.0.0.1:30003")
MODEL = "/data/models/Falcon-H1-7B-Instruct-GPTQ-Int4"
OUT = Path(os.environ["S4_OUT"])
if not str(OUT).startswith("/hy-tmp/sglang-logs/s4-v100-") or OUT.exists():
    raise RuntimeError(f"unsafe or existing S4_OUT: {OUT}")
OUT.parent.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, local_files_only=True, trust_remote_code=True
)
question = tokenizer.encode(
    "\nQuestion: What is 6 times 7? Answer with the number only:",
    add_special_tokens=False,
)
filler = tokenizer.encode("Context filler sentence. ", add_special_tokens=False)


def prompt_ids(length):
    prefix = []
    while len(prefix) + len(question) < length:
        prefix.extend(filler)
    return (prefix[: length - len(question)] + question)[:length]


def snapshot():
    data = requests.get(BASE + "/server_info", timeout=10).json()
    state = data["internal_states"][0]
    hist = {int(k): int(v) for k, v in state.get("decode_batch_hist", {}).items()}
    return hist, state.get("memory_usage", {})


def one(ids, output_length, barrier):
    started = time.perf_counter()
    barrier.wait()
    output_ids, arrivals, error = [], [], None
    try:
        with requests.post(
            BASE + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": output_length,
                    "ignore_eos": True,
                },
                "stream": True,
            },
            stream=True,
            timeout=1800,
        ) as response:
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:400]}"
            else:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    values = json.loads(payload).get("output_ids", [])
                    # Frozen tokenizer manager emits cumulative IDs when incremental
                    # streaming is disabled; replace and timestamp only newly added IDs.
                    if values and len(values) >= len(output_ids):
                        delta = len(values) - len(output_ids)
                        output_ids = list(values)
                        arrivals.extend([time.perf_counter()] * delta)
    except Exception as exc:  # record, then fail wave-level correctness
        error = repr(exc)

    intervals = [arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))]
    return {
        "output_ids": output_ids,
        "output_count": len(output_ids),
        "nonzero_count": sum(bool(token_id) for token_id in output_ids),
        "ttft_s": arrivals[0] - started if arrivals else None,
        "itl_mean_s": statistics.mean(intervals) if intervals else None,
        "itl_p95_s": sorted(intervals)[int(0.95 * (len(intervals) - 1))]
        if intervals
        else None,
        "error": error,
    }


def wave(concurrency, repeat, prompt_length=1024, output_length=256):
    ids = prompt_ids(prompt_length)
    barrier = threading.Barrier(concurrency)
    before_hist, _ = snapshot()
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(
            pool.map(lambda _: one(ids, output_length, barrier), range(concurrency))
        )
    elapsed = time.perf_counter() - started
    after_hist, memory = snapshot()
    batch_hist = {
        size: after_hist.get(size, 0) - before_hist.get(size, 0)
        for size in sorted(set(before_hist) | set(after_hist))
    }
    intervals = [
        row["itl_mean_s"] for row in rows if row["itl_mean_s"] is not None
    ]
    p95_intervals = [
        row["itl_p95_s"] for row in rows if row["itl_p95_s"] is not None
    ]
    correct = all(
        row["output_count"] == output_length
        and row["nonzero_count"] == output_length
        and row["error"] is None
        for row in rows
    )
    record = {
        "concurrency": concurrency,
        "repeat": repeat,
        "prompt_tokens": prompt_length,
        "requested_output_tokens": output_length,
        "wall_s": elapsed,
        "aggregate_tps": sum(row["output_count"] for row in rows) / elapsed,
        "per_request_tps": sum(row["output_count"] for row in rows)
        / (concurrency * elapsed),
        "mean_itl_ms": 1000 * statistics.mean(intervals) if intervals else None,
        "mean_request_p95_itl_ms": 1000 * statistics.mean(p95_intervals)
        if p95_intervals
        else None,
        "correct": correct,
        "decode_batch_histogram": batch_hist,
        "memory_usage_gb": memory,
        "request_errors": [row["error"] for row in rows if row["error"]],
    }
    with OUT.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")
        output.flush()
    print(json.dumps(record), flush=True)
    if not correct:
        raise RuntimeError(
            f"correctness failed: c={concurrency}, repeat={repeat}; record={record}"
        )
    return record


if __name__ == "__main__":
    concurrency = int(os.environ["S4_C"])
    start_repeat = int(os.environ.get("S4_START", "1"))
    for repeat in range(start_repeat, 4):
        wave(concurrency, repeat)

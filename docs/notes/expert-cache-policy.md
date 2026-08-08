# Expert-cache replacement policy lab

This note records an offline experiment only. It does **not** claim a measured runtime
speedup and it does not change the C cache policy.

## Why revisit LRU

The released expert trace contains 100,096 routed-expert requests and 10,010 distinct
experts. At capacities from 8 to 64 GB, the existing LRU replay is flat at 36.24% hit
rate even though the offline Belady ceiling rises from 39.42% to 61.74%. The question is
whether a realistic online policy can recover a useful part of that gap without knowing
the future.

`tools/sim_cache.py` now compares three online candidates against the existing LRU:

- **2Q**: a probation FIFO plus a main LRU queue, intended to resist one-hit scans.
- **ARC**: adaptive recency/frequency balancing with resident and ghost queues.
- **TinyLFU-style admission**: decayed online frequency decides whether a miss replaces
  the current LRU victim.

Belady and PIN+LRU remain future-informed upper bounds. They are not deployable policies.
The Belady implementation was changed from an O(n * capacity) victim scan to an
O(n log capacity) heap replay so the full trace can run cheaply in CI.

## Recorded-trace result

CI replay on `tests/fixtures/expert_trace.bin`:

| Cache | LRU | 2Q | ARC | TinyLFU | Belady | PIN+LRU | Best online |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 GB | 36.24% | 36.23% | 36.14% | 7.69% | 39.42% | 37.82% | LRU |
| 16 GB | 36.24% | 36.24% | 36.14% | 12.95% | 42.61% | 39.42% | LRU |
| 32 GB | 36.24% | 36.62% | 38.33% | 19.70% | 48.99% | 42.57% | ARC |
| 64 GB | 36.24% | 37.20% | **52.65%** | 36.51% | 61.74% | 48.66% | **ARC** |
| 128 GB | 49.19% | 58.99% | 55.64% | **73.03%** | 84.59% | 62.86% | **TinyLFU** |
| 192 GB | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | 90.00% | tie |

At 64 GB, ARC closes 64.4% of the LRU-to-Belady gap. On this trace that changes the
expert-I/O estimate from about 16.47 GB/token under LRU to 12.23 GB/token under ARC,
a reduction of roughly 25.7% in expert bytes read. This is an I/O-model result, not a
wall-clock benchmark.

At 128 GB, TinyLFU closes 67.3% of the LRU-to-Belady gap and estimates 6.97 GB/token,
but it is poor at small capacities. There is therefore no evidence for one universal
replacement policy across all memory budgets.

## Batch-prefetch ordering check

The runtime calls `getmany(top-k)` before consuming the same top-k with `get()`. That can,
in principle, make ordinary LRU evict an old resident that belongs to the current top-k
while reserving slots for other misses. The simulator now models this ordering and also a
variant that protects current-top-k residents during prefetch.

| Cache | Current disk-avoid rate | Protected | Reads saved | Same-top-k self-evictions |
|---:|---:|---:|---:|---:|
| 8 GB | 36.24% | 36.24% | 0 | 0 |
| 16 GB | 36.24% | 36.24% | 0 | 0 |
| 32 GB | 36.24% | 36.24% | 0 | 0 |
| 64 GB | 36.24% | 36.24% | 0 | 0 |
| 128 GB | 49.18% | 49.24% | 63 | 62 |
| 192 GB | 90.00% | 90.00% | 0 | 0 |

So batch self-eviction exists, but it is negligible on this trace and cannot explain the
large 64 GB ARC result.

## What this does *not* prove

The trace predates v1.0's chunk-union prefill and was recorded during a workload that
re-prefilled growing prefixes. Its 90% repeat fraction is therefore not a prediction of
steady-state incremental decode. A runtime policy must be opt-in and measured again on a
v1.0 trace before becoming a default.

The next experiment should implement the strongest candidate behind a one-binary A/B
switch, preserve LRU as the default, and compare byte-identical output, physical expert
bytes read, cache hit accounting, and wall time on the same checkpoint/workload.

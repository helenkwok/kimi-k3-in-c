# ARC runtime A/B experiment

This is an **experimental measurement path**, not a new default cache policy.

The offline policy lab in `expert-cache-policy.md` found that ARC raises the recorded
64 GB trace from 36.24% to 52.65% hit rate, but that trace predates v1.0 chunk-union
prefill. The only useful next question is whether the advantage survives a current
checkpoint run.

## Why ARC is serial-only first

The production cache calls `getmany(top-k)` before consuming the same top-k with `get()`.
That reservation order is deliberately concurrent and has different timing from the
logical request stream used by the policy replay. Mixing a new replacement policy into
that path at the same time would change two things at once.

`K3_CACHE_POLICY=arc` therefore disables batch prefetch automatically. The fair baseline
also disables it:

```bash
K3_NOPREFETCH=1 K3_CACHE_POLICY=lru ./bin/k3 <same arguments>
K3_CACHE_POLICY=arc                  ./bin/k3 <same arguments>
```

Use the **same binary, prompt/ids, memory budgets, thread count and checkpoint** for both
runs. ARC is not being compared with normal production LRU here; it is being compared
with serial LRU so replacement policy is the intended independent variable.

## What to record

For each side of the A/B, record:

1. emitted token ids (must be identical),
2. `cache [...]` requests/hits/misses/evictions,
3. `read from disk` bytes and loading time,
4. total token wall time,
5. `--dump-cache-trace` output for replay through `tools/sim_cache.py`.

A useful first sweep is the region where the old trace showed separation: roughly 32,
64 and 128 GB of expert cache. Repeat timings; the repository's earlier campaign found
enough run-to-run variation that a single timing should not be treated as a speedup.

## Safety behavior

ARC metadata owns no expert bytes. Physical expert buffers still live in `K3Cache` and
are validated by the existing streaming-cache fixture.

The experiment falls back to ordinary LRU if:

- ARC logical residency ever disagrees with the physical slot map,
- ARC selects a physical victim that is unavailable,
- a read fails after ARC has changed its metadata, or
- permanent pinning is requested.

Unknown `K3_CACHE_POLICY` values are rejected rather than silently interpreted as LRU.

## Promotion criteria

Do **not** make ARC the default from the old trace alone. A later production integration
should require all of the following on a v1 workload:

- byte-identical model output,
- materially fewer physical expert bytes read at a useful memory tier,
- a replicated wall-clock improvement larger than measurement noise,
- no regression at low-memory tiers where the offline trace showed LRU already winning,
- a separate design/test for ARC plus concurrent `getmany` before enabling both together.

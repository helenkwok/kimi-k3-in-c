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

## Reproducible harness

`benchmarks/cache-policy-ab.sh` runs the comparison, alternates policy order between
repetitions, verifies generated ids, saves a cache trace for every run, and prints a
summary of whole-run physical expert I/O and timing dispersion.

The first useful run targets the 64 GB expert-cache tier where the old trace showed the
largest ARC separation:

```bash
make -j
benchmarks/cache-policy-ab.sh ~/k3model ~/k3trunk /tmp/k3-arc-ab 64 60 3
```

Arguments are:

```text
cache-policy-ab.sh <model_dir> <trunk_dir> <out_dir> [cache_gb_csv] [trunk_gb] [reps]
```

A broader sweep, on a machine with enough RAM, is:

```bash
benchmarks/cache-policy-ab.sh ~/k3model ~/k3trunk /tmp/k3-arc-ab \
    32,64,128 60 3
```

The harness uses token ids by default so tokenizer behavior cannot confound the A/B. Set
`K3_AB_IDS`, `K3_AB_GEN`, and `OMP_NUM_THREADS` before invoking it to change the workload
while keeping both policies identical.

It deliberately alternates order: odd repetitions run LRU then ARC, even repetitions run
ARC then LRU. Even with direct I/O, device temperature and unrelated system load can drift
over a long experiment, so always running one policy second would confound policy with
order.

## What the harness records

For each side of the A/B it records:

1. emitted token ids (**hard gate: they must be identical**),
2. `seconds_per_token` from the run JSON,
3. the CLI's `experts, whole run: ... GB read` physical expert-I/O total,
4. peak RSS,
5. raw logs and JSON results,
6. `--dump-cache-trace` output for replay through `tools/sim_cache.py`.

The physical-I/O total is intentionally the primary policy metric. The final-step raw
cache hit count is not suitable: batch prefetch can count an expert as a hit even when it
was read from disk moments earlier. The CLI already accumulates the whole-run expert bytes
across generation steps, and the harness parses that explicit total.

Timing is secondary and must be replicated. The summary prints mean, standard deviation,
and within-policy spread, then compares the apparent ARC speed gain against that spread.
A negative result is valid: the harness fails on changed output or missing measurements,
not merely because ARC is slower or reads more bytes.

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

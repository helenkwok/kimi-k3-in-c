#!/usr/bin/env python
"""
sim_cache.py - compare expert-cache replacement policies on a recorded K3 trace.

THE IDEA
    Routing does not depend on cache replacement, so one inference run can record the
    entire (layer, expert) request stream and this tool can replay it at many capacities.

POLICIES
    LRU       what the C engine implements today.
    2Q        online, scan-resistant: first touches enter a small probation FIFO; a
              second touch promotes the expert to a main LRU queue.
    ARC       online and self-tuning: balances recency and frequency using resident and
              ghost queues, with no trace-specific parameter fitting.
    TinyLFU   online admission control: keep LRU replacement, but bypass a one-hit
              candidate when its decayed frequency is no better than the LRU victim.
    Belady    offline optimum. It sees the future and is only an upper bound.
    PIN+LRU   hottest-half pinning chosen from this same trace. Also future knowledge;
              useful only as an upper bound on profile-guided pinning.

BATCH PREFETCH
    The C engine calls getmany(top-k) before consuming that top-k. The ordinary sequential
    replay therefore misses a second effect: while reserving slots for misses, getmany can
    evict an expert that is already resident and belongs to the SAME top-k, before get()
    has touched it and refreshed its LRU stamp. batch_lru() models this ordering exactly
    enough to count disk reads, and a protected variant excludes the current top-k from
    prefetch victims. That protection is an independent runtime opportunity from changing
    the replacement policy itself.

The online policies are deliberately simulation-only here. A policy should first close a
meaningful part of the LRU-to-Belady gap on recorded traces before runtime C is changed.

usage:
    sim_cache.py expert_trace.bin
    sim_cache.py expert_trace.bin --caps-gb 8,32,64,128,192
"""
from __future__ import annotations

import argparse
import heapq
import sys
from collections import Counter, OrderedDict

import numpy as np

EXPERT_BYTES = 17_547_264
TOTAL_EXPERTS = 92 * 896
DEFAULT_CAPS_GB = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1450]


def lru(trace, cap):
    """Classic sequential LRU. Returns hit count."""
    if cap <= 0:
        return 0
    seen = OrderedDict()
    hits = 0
    for key in trace:
        if key in seen:
            seen.move_to_end(key)
            hits += 1
        else:
            if len(seen) >= cap:
                seen.popitem(last=False)
            seen[key] = None
    return hits


def twoq(trace, cap, in_ratio=0.25, ghost_ratio=1.0):
    """A compact online 2Q variant.

    New entries enter A1in, a small FIFO. A second resident touch, or a return after
    eviction into the A1out ghost queue, promotes the entry to Am, an LRU queue. Long
    one-touch scans therefore churn A1in rather than flushing the frequent set.
    """
    if cap <= 0:
        return 0
    if cap == 1:
        return lru(trace, cap)

    kin = min(cap - 1, max(1, round(cap * in_ratio)))
    kout = max(1, round(cap * ghost_ratio))
    a1in = OrderedDict()
    am = OrderedDict()
    a1out = OrderedDict()
    hits = 0

    def ghost_add(key):
        a1out[key] = None
        a1out.move_to_end(key)
        while len(a1out) > kout:
            a1out.popitem(last=False)

    def evict_for_new():
        if len(a1in) >= kin and a1in:
            old, _ = a1in.popitem(last=False)
            ghost_add(old)
        elif am:
            am.popitem(last=False)
        elif a1in:
            old, _ = a1in.popitem(last=False)
            ghost_add(old)

    for key in trace:
        if key in am:
            am.move_to_end(key)
            hits += 1
            continue

        if key in a1in:
            hits += 1
            a1in.pop(key)
            am[key] = None
            continue

        if key in a1out:
            a1out.pop(key)
            if len(a1in) + len(am) >= cap:
                evict_for_new()
            am[key] = None
            continue

        if len(a1in) + len(am) >= cap:
            evict_for_new()
        a1in[key] = None

    return hits


def arc(trace, cap):
    """Adaptive Replacement Cache (ARC), implemented from the published algorithm.

    T1/T2 are resident recency/frequency queues; B1/B2 are key-only ghost queues. The
    target size p moves online in response to ghost hits, so there is no trace-specific
    recency/frequency split to tune.
    """
    if cap <= 0:
        return 0

    t1 = OrderedDict()
    t2 = OrderedDict()
    b1 = OrderedDict()
    b2 = OrderedDict()
    p = 0.0
    hits = 0

    def trim_ghost(queue):
        while len(queue) > cap:
            queue.popitem(last=False)

    def replace(incoming):
        prefer_t1 = bool(t1) and (
            (incoming in b2 and len(t1) >= p) or len(t1) > p
        )
        if prefer_t1 or not t2:
            if t1:
                old, _ = t1.popitem(last=False)
                b1[old] = None
                trim_ghost(b1)
        else:
            old, _ = t2.popitem(last=False)
            b2[old] = None
            trim_ghost(b2)

    for key in trace:
        if key in t1:
            hits += 1
            t1.pop(key)
            t2[key] = None
            continue

        if key in t2:
            hits += 1
            t2.move_to_end(key)
            continue

        if key in b1:
            delta = 1.0 if len(b1) >= len(b2) else len(b2) / max(len(b1), 1)
            p = min(float(cap), p + delta)
            replace(key)
            b1.pop(key, None)
            t2[key] = None
            continue

        if key in b2:
            delta = 1.0 if len(b2) >= len(b1) else len(b1) / max(len(b2), 1)
            p = max(0.0, p - delta)
            replace(key)
            b2.pop(key, None)
            t2[key] = None
            continue

        if len(t1) + len(b1) == cap:
            if len(t1) < cap:
                if b1:
                    b1.popitem(last=False)
                replace(key)
            elif t1:
                t1.popitem(last=False)
        elif len(t1) + len(b1) < cap:
            total = len(t1) + len(t2) + len(b1) + len(b2)
            if total >= cap:
                if total >= 2 * cap and b2:
                    b2.popitem(last=False)
                replace(key)

        t1[key] = None

        if len(t1) + len(t2) > cap:
            raise AssertionError("ARC resident set exceeded capacity")

    return hits


def tinylfu(trace, cap, sample_factor=10):
    """TinyLFU-style online admission in front of an LRU resident set.

    Frequencies are learned only from requests already observed. They are halved every
    sample_factor * capacity requests so an early phase cannot dominate forever. On a
    miss with a full cache, the candidate is admitted only if its frequency is strictly
    greater than the current LRU victim's; otherwise the read is served but not cached.
    """
    if cap <= 0:
        return 0

    resident = OrderedDict()
    freq = Counter()
    hits = 0
    sample = max(1, sample_factor * cap)
    since_age = 0

    for key in trace:
        freq[key] += 1
        since_age += 1

        if key in resident:
            hits += 1
            resident.move_to_end(key)
        elif len(resident) < cap:
            resident[key] = None
        else:
            victim = next(iter(resident))
            if freq[key] > freq[victim]:
                resident.popitem(last=False)
                resident[key] = None

        if since_age >= sample:
            for old in list(freq):
                freq[old] //= 2
                if freq[old] == 0:
                    del freq[old]
            since_age = 0

    return hits


def belady(trace, cap):
    """Optimal offline replacement, O(n log cap)."""
    if cap <= 0:
        return 0

    n = len(trace)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        key = trace[i]
        nxt[i] = last.get(key, n)
        last[key] = i

    resident = {}
    farthest = []
    hits = 0

    for i, key in enumerate(trace):
        next_use = nxt[i]
        if key in resident:
            hits += 1
            resident[key] = next_use
            heapq.heappush(farthest, (-next_use, key))
            continue

        if len(resident) >= cap:
            while farthest:
                neg_next, victim = farthest[0]
                if victim in resident and resident[victim] == -neg_next:
                    break
                heapq.heappop(farthest)

            victim_next = -farthest[0][0]
            if victim_next < next_use:
                continue

            _, victim = heapq.heappop(farthest)
            del resident[victim]

        resident[key] = next_use
        heapq.heappush(farthest, (-next_use, key))

    return hits


def pinned_lru(trace, cap, npin):
    """Future-informed upper bound: pin the hottest keys from this same trace."""
    if cap <= 0:
        return 0
    if npin >= cap:
        npin = cap - 1

    hot = {key for key, _ in Counter(trace).most_common(max(npin, 0))}
    loaded = set()
    seen = OrderedDict()
    room = max(cap - len(hot), 1)
    hits = 0

    for key in trace:
        if key in hot:
            if key in loaded:
                hits += 1
            else:
                loaded.add(key)
            continue

        if key in seen:
            seen.move_to_end(key)
            hits += 1
        else:
            if len(seen) >= room:
                seen.popitem(last=False)
            seen[key] = None

    return hits


def make_batches(pairs, topk=16):
    """Split recorded (layer, expert) pairs into the getmany groups used by the engine.

    A routed top-k contains one layer and at most topk consecutive expert requests. The
    cap matters because repeated prefill can place the same layer consecutively for more
    than one token; those are separate getmany calls, not one giant batch.
    """
    batches = []
    batch = []
    layer0 = None
    for layer, expert in pairs:
        layer = int(layer)
        expert = int(expert)
        if batch and (layer != layer0 or len(batch) >= topk):
            batches.append(batch)
            batch = []
        if not batch:
            layer0 = layer
        batch.append((layer << 20) | expert)
    if batch:
        batches.append(batch)
    return batches


def batch_lru(pairs, cap, topk=16, protect_current=False):
    """Replay the C cache's getmany-then-get ordering and count physical reads.

    In the prefetch phase, miss slots are reserved while ordinary resident entries have
    NOT yet had their LRU timestamps refreshed by get(). With protect_current=False this
    models today's cache: an old resident belonging to the current top-k can therefore be
    selected as a victim and need a second synchronous read moments later.

    protect_current=True excludes entries in the current top-k from prefetch victims.
    It uses no future knowledge beyond the top-k ids getmany already receives.
    """
    if cap <= 0:
        return 0, 0, 0

    resident = OrderedDict()
    reads = 0
    self_evictions = 0
    requests = 0

    for batch in make_batches(pairs, topk=topk):
        requests += len(batch)
        current = set(batch)
        inflight = []

        # Phase 1: reserve a distinct slot for every miss. Inflight slots are not victim
        # candidates, matching K3_SLOT_INFLIGHT in the C cache.
        for key in batch:
            if key in resident or key in inflight:
                continue

            if len(resident) + len(inflight) >= cap:
                victim = None
                for candidate in resident:
                    if protect_current and candidate in current:
                        continue
                    victim = candidate
                    break
                if victim is None:
                    # topk+1 capacity should make this unreachable for the released
                    # model. Keep the simulator total rather than invent a victim.
                    continue
                if victim in current:
                    self_evictions += 1
                del resident[victim]

            inflight.append(key)
            reads += 1

        # Phase 3 publish: reservation order is the used_at order assigned by getmany.
        for key in inflight:
            resident[key] = None

        # Consume the top-k. A resident evicted during prefetch has to be read again.
        for key in batch:
            if key in resident:
                resident.move_to_end(key)
                continue

            reads += 1
            if len(resident) >= cap:
                resident.popitem(last=False)
            resident[key] = None

    return requests, reads, self_evictions


ONLINE_POLICIES = {
    "LRU": lru,
    "2Q": twoq,
    "ARC": arc,
    "TinyLFU": tinylfu,
}


def parse_caps(text):
    if not text:
        return DEFAULT_CAPS_GB
    try:
        caps = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "caps must be comma-separated integer GB values"
        ) from exc
    if not caps or any(gb <= 0 for gb in caps):
        raise argparse.ArgumentTypeError("caps must contain positive integer GB values")
    return caps


def pct(hits, n):
    return 100.0 * hits / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    ap.add_argument(
        "--disk-mbs",
        type=float,
        default=1234.0,
        help="measured random-cold read rate, for the time column",
    )
    ap.add_argument(
        "--caps-gb",
        default=None,
        help="comma-separated capacities to replay (default: historical campaign set)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=16,
        help="getmany batch width for the batch-prefetch replay (default: 16)",
    )
    args = ap.parse_args()
    caps_gb = parse_caps(args.caps_gb)

    raw = np.fromfile(args.trace, dtype=np.int32)
    if raw.size % 2:
        sys.exit("trace is not an even number of int32")
    pairs = raw.reshape(-1, 2)
    layers, experts = pairs[:, 0], pairs[:, 1]
    keys = (layers.astype(np.int64) << 20) | experts.astype(np.int64)
    trace = keys.tolist()

    n = len(trace)
    uniq = len(set(trace))
    per_tok = 1472
    ntok = max(n // per_tok, 1)
    print(f"trace: {n} requests, {uniq} distinct experts, about {ntok} token(s)")
    print(
        "distinct experts touched: %d of %d (%.2f%% of the pool)"
        % (uniq, TOTAL_EXPERTS, 100.0 * uniq / TOTAL_EXPERTS)
    )
    print(
        "if nothing were cached: %.2f GB per token\n"
        % (per_tok * args.expert_bytes / 1e9)
    )

    counts = Counter(trace)
    top = counts.most_common(10)
    print(
        "hottest experts: "
        + ", ".join(
            "L%d/e%d x%d" % (key >> 20, key & 0xFFFFF, count)
            for key, count in top[:6]
        )
    )
    reuse = sum(count - 1 for count in counts.values())
    print(
        "total reuse: %d of %d requests are repeats (%.1f%%)\n"
        % (reuse, n, 100.0 * reuse / n)
    )

    header = (
        f"{'CACHE':<9} {'SLOTS':>8} {'LRU':>9} {'2Q':>9} {'ARC':>9} "
        f"{'TinyLFU':>9} {'BELADY':>9} {'PIN+LRU':>9} {'BEST':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for gb in caps_gb:
        cap = int(gb * 1e9 // args.expert_bytes)
        if cap < 1:
            continue

        online = {name: fn(trace, cap) for name, fn in ONLINE_POLICIES.items()}
        optimum = belady(trace, cap)
        pinned = pinned_lru(trace, cap, cap // 2)
        best_name, best_hits = max(online.items(), key=lambda item: item[1])

        rows.append((gb, cap, online, optimum, pinned, best_name, best_hits))
        print(
            f"{gb:<7d}GB {cap:8d} "
            f"{pct(online['LRU'], n):8.2f}% {pct(online['2Q'], n):8.2f}% "
            f"{pct(online['ARC'], n):8.2f}% {pct(online['TinyLFU'], n):8.2f}% "
            f"{pct(optimum, n):8.2f}% {pct(pinned, n):8.2f}% "
            f"{best_name:>10}"
        )

    print("-" * len(header))
    print("\nOnline-policy value relative to LRU:")
    for gb, _cap, online, optimum, _pinned, best_name, best_hits in rows:
        base = online["LRU"]
        gap = optimum - base
        closed = 100.0 * (best_hits - base) / gap if gap > 0 else 0.0
        misses = n - best_hits
        gb_tok = misses * args.expert_bytes / 1e9 / ntok
        sec = gb_tok * 1000.0 / args.disk_mbs
        print(
            f"  {gb:4d} GB: {best_name:<7} {pct(best_hits, n):6.2f}% hit, "
            f"closes {closed:5.1f}% of LRU->Belady gap; "
            f"{gb_tok:6.2f} GB/tok, {sec:6.2f} s/tok expert I/O"
        )

    print("\nBatch-prefetch ordering (physical disk reads):")
    print(
        f"{'CACHE':<9} {'CURRENT':>10} {'PROTECTED':>10} {'SAVED':>9} "
        f"{'SELF-EVICT':>11}"
    )
    print("-" * 55)
    for gb, cap, *_rest in rows:
        req, reads_now, self_evict = batch_lru(
            pairs, cap, topk=args.topk, protect_current=False
        )
        _req2, reads_protected, protected_evict = batch_lru(
            pairs, cap, topk=args.topk, protect_current=True
        )
        saved = reads_now - reads_protected
        current_rate = 100.0 * (req - reads_now) / req if req else 0.0
        protected_rate = 100.0 * (req - reads_protected) / req if req else 0.0
        print(
            f"{gb:<7d}GB {current_rate:9.2f}% {protected_rate:9.2f}% "
            f"{saved:9d} {self_evict:11d}"
        )
        if protected_evict:
            raise AssertionError("protected batch replay self-evicted a current top-k key")

    print(
        "\nCURRENT/PROTECTED are disk-avoidance rates, not raw cache.hits: every physical "
        "expert read is charged, including a reload caused by self-eviction during getmany."
    )
    print(
        "Belady and PIN+LRU use future knowledge and are ceilings, not deployable policies."
    )
    print("2Q, ARC and TinyLFU are online: each decision uses only requests seen so far.")
    comp = uniq
    print(
        "compulsory misses: %d; absolute hit-rate ceiling on this trace is %.2f%%."
        % (comp, 100.0 * (n - comp) / n)
    )

    full = uniq * args.expert_bytes / 1e9
    print("\nholding every expert this trace touched needs %.2f GB." % full)
    print(
        "holding the whole pool needs %.2f GB."
        % (TOTAL_EXPERTS * args.expert_bytes / 1e9)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

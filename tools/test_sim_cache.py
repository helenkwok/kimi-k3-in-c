#!/usr/bin/env python
"""Deterministic policy tests for tools/sim_cache.py; no checkpoint required."""
from __future__ import annotations

import random

import sim_cache as s


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    trace = [1, 2, 1, 3, 1, 2]
    check(s.lru(trace, 2) == 2, "LRU reference trace changed")
    check(s.belady(trace, 2) == 3, "Belady reference trace changed")

    unique = list(range(100))
    for name, policy in s.ONLINE_POLICIES.items():
        check(policy(unique, 8) == 0, f"{name} invented hits on a unique scan")

    repeated = [7] * 20
    for name, policy in s.ONLINE_POLICIES.items():
        check(policy(repeated, 1) == 19, f"{name} mishandled a one-slot hot key")

    # Prime two hot entries with second touches, run a one-hit scan larger than the
    # cache, then ask for the hot pair again. Plain LRU loses them; scan-resistant
    # policies should preserve enough frequency/history to do better.
    scan = [0, 1, 0, 1] + list(range(2, 20)) + [0, 1]
    base = s.lru(scan, 4)
    check(base == 2, "scan fixture no longer distinguishes LRU")
    check(s.twoq(scan, 4) > base, "2Q failed the scan-resistance fixture")
    check(s.arc(scan, 4) > base, "ARC failed the scan-resistance fixture")
    check(s.tinylfu(scan, 4) > base, "TinyLFU failed the scan-resistance fixture")

    # Model two top-4 getmany calls at capacity 5. Expert 0 is an old resident needed by
    # the second top-k. Today's prefetch can evict it while reserving 4/5/6, then get(0)
    # must read it again. Protecting the current top-k removes that redundant read using
    # information getmany already has.
    pairs = [(0, e) for e in [0, 1, 2, 3, 0, 4, 5, 6]]
    batches = s.make_batches(pairs, topk=4)
    check(len(batches) == 2 and all(len(b) == 4 for b in batches), "batch split failed")
    req, reads_now, self_evict = s.batch_lru(pairs, 5, topk=4, protect_current=False)
    req2, reads_protected, protected_evict = s.batch_lru(
        pairs, 5, topk=4, protect_current=True
    )
    check(req == req2 == 8, "batch request accounting changed")
    check(self_evict > 0, "batch fixture no longer triggers self-eviction")
    check(reads_protected < reads_now, "top-k protection did not save a disk read")
    check(protected_evict == 0, "protected batch still self-evicted")

    # No online implementation may beat the offline optimum. Exercise many small,
    # deterministic traces so ARC/2Q queue bookkeeping is tested beyond hand examples.
    rng = random.Random(0x4B33)
    for cap in range(1, 9):
        for _ in range(20):
            random_trace = [rng.randrange(12) for _ in range(80)]
            optimum = s.belady(random_trace, cap)
            for name, policy in s.ONLINE_POLICIES.items():
                hits = policy(random_trace, cap)
                check(0 <= hits <= optimum, f"{name} exceeded Belady at cap {cap}")

    check(s.parse_caps(None) == s.DEFAULT_CAPS_GB, "default capacities changed")
    check(s.parse_caps("8,64,128") == [8, 64, 128], "capacity parser failed")

    print("CACHE POLICY TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

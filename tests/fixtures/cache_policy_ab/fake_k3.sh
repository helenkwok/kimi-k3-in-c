#!/usr/bin/env bash
# Fake k3 used only to exercise benchmarks/cache-policy-ab.sh in CI.
set -euo pipefail

if [ "${1:-}" = "--version" ]; then
    echo "k3 fake-cache-policy-ab"
    exit 0
fi

policy="${K3_CACHE_POLICY:-lru}"
if [ "$policy" = "lru" ]; then
    [ "${K3_NOPREFETCH:-}" = "1" ] || { echo "fake: LRU baseline did not disable prefetch" >&2; exit 9; }
elif [ "$policy" = "arc" ]; then
    if [ "${K3_NOPREFETCH+x}" = x ]; then
        echo "fake: ARC inherited K3_NOPREFETCH" >&2
        exit 9
    fi
else
    echo "fake: unexpected policy $policy" >&2
    exit 9
fi

out=""
trace=""
cache=""
gen=4
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --dump-cache-trace) trace="$2"; shift 2 ;;
        --cache-gb) cache="$2"; shift 2 ;;
        --gen) gen="$2"; shift 2 ;;
        *) shift ;;
    esac
done

[ -n "$out" ] && [ -n "$trace" ] && [ -n "$cache" ] || {
    echo "fake: missing output, trace or cache argument" >&2
    exit 9
}

case "$policy:$cache" in
    lru:32) spt=12.0; gb=18.0 ;;
    arc:32) spt=11.5; gb=16.5 ;;
    lru:64) spt=10.0; gb=16.47 ;;
    arc:64) spt=8.0;  gb=12.23 ;;
    lru:*)  spt=10.0; gb=15.0 ;;
    arc:*)  spt=9.0;  gb=13.0 ;;
esac

mkdir -p "$(dirname "$out")" "$trace"
printf '{"prompt_ids":[1,2],"generated_ids":[11,22,33,44],"full_ids":[1,2,11,22,33,44],"layers":93,"seconds_per_token":%s}\n' "$spt" > "$out"
printf 'trace\n' > "$trace/expert_trace.bin"
printf '{}\n' > "$trace/expert_hist.json"

printf '%s tokens in 40.0 s, %s s/token average\n' "$gen" "$spt"
printf 'PEAK RSS for the whole run: 100.00 GB   <- quote this, not the plan\n'
printf '  experts, whole run: %s GB read | 100 of 200 requests retained in RAM (50.00%%) | 100 evictions\n' "$gb"

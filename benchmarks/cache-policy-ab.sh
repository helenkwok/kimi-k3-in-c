#!/usr/bin/env bash
# Compare serial LRU against experimental ARC on the SAME binary and checkpoint.
#
#   benchmarks/cache-policy-ab.sh <model_dir> <trunk_dir> <out_dir> \
#       [cache_gb_csv] [trunk_gb] [reps]
#
# Defaults target the strongest separation in the historical trace: a 64 GB expert
# cache, with enough trunk pinned to avoid making trunk I/O dominate the experiment.
# Override the cache list (for example 32,64,128) to sweep capacity.
#
# This harness does NOT decide that ARC is good. It establishes whether it is good on a
# current checkpoint. Correctness is a hard gate (generated ids must match); I/O and
# timing are measurements and may legitimately show no benefit or a regression.
set -euo pipefail

MODEL="${1:?usage: cache-policy-ab.sh <model_dir> <trunk_dir> <out_dir> [cache_gb_csv] [trunk_gb] [reps]}"
TRUNK="${2:?}"
OUT="${3:?}"
CACHE_CSV="${4:-64}"
TRUNK_GB="${5:-60}"
REPS="${6:-3}"
K3_BIN="${K3_BIN:-./bin/k3}"
IDS="${K3_AB_IDS:-1008,10484,318,15383,387}"
GEN="${K3_AB_GEN:-8}"

[ -x "$K3_BIN" ] || { echo "$K3_BIN not executable; build bin/k3 first or set K3_BIN"; exit 1; }
[ -d "$MODEL" ] || { echo "no such model dir: $MODEL"; exit 1; }
[ -d "$TRUNK" ] || { echo "no such trunk dir: $TRUNK"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for JSON checks and summary"; exit 1; }
case "$REPS" in ''|*[!0-9]*) echo "reps must be a positive integer"; exit 1;; esac
[ "$REPS" -ge 1 ] || { echo "reps must be >= 1"; exit 1; }

IFS=',' read -r -a CACHES <<< "$CACHE_CSV"
[ "${#CACHES[@]}" -gt 0 ] || { echo "cache_gb_csv is empty"; exit 1; }
for cache in "${CACHES[@]}"; do
    [[ "$cache" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        echo "invalid cache size '$cache'; use comma-separated GB numbers"; exit 1;
    }
done

mkdir -p "$OUT"
TSV="$OUT/cache-policy-ab.tsv"
printf 'cache_gb\ttrunk_gb\trep\torder\tpolicy\ts_per_tok\texpert_gb\tpeak_rss_gb\tids\n' > "$TSV"

{
    echo "date          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host          : $(uname -srm)"
    echo "cpu           : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//' || true)"
    echo "cores         : $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
    echo "memtotal      : $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo 2>/dev/null || true)"
    echo "model_fs      : $(df -PT "$MODEL" 2>/dev/null | awk 'NR==2{print $2, $1}' || true)"
    echo "trunk_fs      : $(df -PT "$TRUNK" 2>/dev/null | awk 'NR==2{print $2, $1}' || true)"
    echo "k3            : $($K3_BIN --version 2>&1 | head -1)"
    echo "cache_gb      : $CACHE_CSV"
    echo "trunk_gb      : $TRUNK_GB"
    echo "reps/policy   : $REPS"
    echo "OMP threads   : ${OMP_NUM_THREADS:-default}"
    echo "ids           : $IDS"
    echo "gen           : $GEN"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"
echo

REF_IDS=""
REF_TAG=""

run_one() {
    local cache="$1" rep="$2" order="$3" policy="$4"
    local tag="cache${cache}_r${rep}_${policy}"
    local json="$OUT/$tag.json"
    local log="$OUT/$tag.log"
    local trace="$OUT/$tag.trace"
    mkdir -p "$trace"

    echo "== cache ${cache} GB | rep ${rep}/${REPS} | ${policy^^} | order ${order} =="

    if [ "$policy" = "lru" ]; then
        env K3_NOPREFETCH=1 K3_CACHE_POLICY=lru \
            "$K3_BIN" "$MODEL" --ids "$IDS" --gen "$GEN" --incremental \
            --trunk "$TRUNK" --trunk-gb "$TRUNK_GB" --cache-gb "$cache" \
            --out "$json" --dump-cache-trace "$trace" > "$log" 2>&1
    else
        # ARC itself disables getmany. Explicitly REMOVE K3_NOPREFETCH from the inherited
        # environment so an outer shell cannot accidentally make the two commands look
        # equivalent for the wrong reason.
        env -u K3_NOPREFETCH K3_CACHE_POLICY=arc \
            "$K3_BIN" "$MODEL" --ids "$IDS" --gen "$GEN" --incremental \
            --trunk "$TRUNK" --trunk-gb "$TRUNK_GB" --cache-gb "$cache" \
            --out "$json" --dump-cache-trace "$trace" > "$log" 2>&1
    fi

    local ids spt expert_gb rss
    ids=$(python3 - "$json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)
ids = data.get("generated_ids")
if not isinstance(ids, list) or not ids:
    raise SystemExit(2)
print(",".join(map(str, ids)))
PY
) || { echo "*** unreadable generated_ids in $json ***"; exit 1; }

    spt=$(python3 - "$json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    value = json.load(f).get("seconds_per_token")
if not isinstance(value, (int, float)) or value <= 0:
    raise SystemExit(2)
print(f"{value:.6f}")
PY
) || { echo "*** unreadable seconds_per_token in $json ***"; exit 1; }

    # This is the whole-run physical expert traffic accumulated across every generation
    # step. Do not use the final cache report's raw hits: batch prefetch counts disk reads
    # as hits when get() consumes them moments later.
    expert_gb=$(grep -oE 'experts, whole run: [0-9.]+ GB read' "$log" | tail -1 | awk '{print $4}')
    [ -n "$expert_gb" ] || {
        echo "*** missing whole-run expert GB metric in $log ***"
        tail -20 "$log"
        exit 1
    }

    rss=$(grep -oE 'PEAK RSS for the whole run: [0-9.]+' "$log" | tail -1 | awk '{print $7}')
    [ -n "$rss" ] || rss="-"

    [ -s "$trace/expert_trace.bin" ] || {
        echo "*** cache trace was not written: $trace/expert_trace.bin ***"; exit 1;
    }

    if [ -z "$REF_IDS" ]; then
        REF_IDS="$ids"; REF_TAG="$tag"
    elif [ "$ids" != "$REF_IDS" ]; then
        echo "*** OUTPUT DIFFERS: replacement policy changed model output ***"
        echo "reference ($REF_TAG): $REF_IDS"
        echo "current   ($tag): $ids"
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$cache" "$TRUNK_GB" "$rep" "$order" "$policy" "$spt" \
        "$expert_gb" "$rss" "$ids" >> "$TSV"
    echo "   ${spt} s/token | ${expert_gb} GB expert reads | peak RSS ${rss} GB"
}

for cache in "${CACHES[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        # Alternate the run order. Even O_DIRECT workloads can see device-temperature,
        # queueing and system-load drift over a long experiment; always running ARC
        # second would confound policy with time/order.
        if [ $((rep % 2)) -eq 1 ]; then
            run_one "$cache" "$rep" 1 lru
            run_one "$cache" "$rep" 2 arc
        else
            run_one "$cache" "$rep" 1 arc
            run_one "$cache" "$rep" 2 lru
        fi
    done
done

echo
if command -v column >/dev/null 2>&1; then column -t -s $'\t' "$TSV"; else cat "$TSV"; fi
echo

python3 - "$TSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
by = defaultdict(lambda: defaultdict(list))
for row in rows:
    by[float(row["cache_gb"])][row["policy"]].append(row)

print(f"{'cache':>9}  {'policy':>7}  {'s/tok mean':>10}  {'sd':>7}  {'spread':>8}  {'expert GB':>10}  n")
print("-" * 72)
for cache in sorted(by):
    for policy in ("lru", "arc"):
        group = by[cache].get(policy, [])
        if not group:
            continue
        times = [float(r["s_per_tok"]) for r in group]
        reads = [float(r["expert_gb"]) for r in group]
        mean_t = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        spread = (max(times) - min(times)) / mean_t * 100 if len(times) > 1 else 0.0
        print(f"{cache:7.1f}GB  {policy:>7}  {mean_t:10.3f}  {sd:7.3f}  {spread:7.1f}%  "
              f"{statistics.mean(reads):10.2f}  {len(group)}")

print("\nPAIRWISE RESULT")
for cache in sorted(by):
    lru = by[cache].get("lru", [])
    arc = by[cache].get("arc", [])
    if not lru or not arc:
        continue
    lt = statistics.mean(float(r["s_per_tok"]) for r in lru)
    at = statistics.mean(float(r["s_per_tok"]) for r in arc)
    lb = statistics.mean(float(r["expert_gb"]) for r in lru)
    ab = statistics.mean(float(r["expert_gb"]) for r in arc)
    speedup = lt / at
    io_reduction = 100.0 * (lb - ab) / lb if lb else 0.0
    spreads = []
    for group in (lru, arc):
        vals = [float(r["s_per_tok"]) for r in group]
        if len(vals) > 1:
            spreads.append((max(vals) - min(vals)) / statistics.mean(vals) * 100)
    noise = max(spreads, default=0.0)
    speed_gain = 100.0 * (lt - at) / lt if lt else 0.0
    print(f"  {cache:7.1f} GB: expert I/O {io_reduction:+6.1f}% | speedup {speedup:5.2f}x "
          f"({speed_gain:+5.1f}%) | worst within-policy spread {noise:5.1f}%")
    if len(lru) == 1 or len(arc) == 1:
        print("             one sample on at least one side: timing is descriptive only")
    elif speed_gain > noise and io_reduction > 0:
        print("             signal clears this run's simple spread check; still replicate on another workload")
    else:
        print("             wall-clock advantage does NOT clearly exceed within-policy spread")

print("\nCorrectness gate: all generated-id sequences were identical.")
print("Primary policy metric: whole-run physical expert GB read; timing is secondary and must clear noise.")
PY

echo
echo "A/B complete. Raw logs, JSON and traces: $OUT"
echo "Machine provenance: $OUT/machine.txt"

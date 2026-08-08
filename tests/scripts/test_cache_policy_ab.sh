#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/model" "$TMP/trunk" "$TMP/out"
cp "$ROOT/tests/fixtures/cache_policy_ab/fake_k3.sh" "$TMP/fake-k3"
chmod +x "$TMP/fake-k3"

K3_BIN="$TMP/fake-k3" \
K3_AB_GEN=4 \
bash "$ROOT/benchmarks/cache-policy-ab.sh" \
    "$TMP/model" "$TMP/trunk" "$TMP/out" 32,64 3 2 > "$TMP/harness.log"

TSV="$TMP/out/cache-policy-ab.tsv"
[ -s "$TSV" ] || { echo "missing A/B TSV"; exit 1; }
# header + 2 capacities * 2 repetitions * 2 policies = 9 lines.
[ "$(wc -l < "$TSV")" -eq 9 ] || { echo "unexpected A/B row count"; cat "$TSV"; exit 1; }

grep -q $'64\t3\t1\t1\tlru\t10.000000\t16.47' "$TSV"
grep -q $'64\t3\t1\t2\tarc\t8.000000\t12.23' "$TSV"
# Rep 2 must reverse the order to guard against a permanently policy-ordered harness.
grep -q $'64\t3\t2\t1\tarc' "$TSV"
grep -q $'64\t3\t2\t2\tlru' "$TSV"

grep -q 'Correctness gate: all generated-id sequences were identical.' "$TMP/harness.log"
grep -q '64.0 GB: expert I/O' "$TMP/harness.log"
grep -q 'A/B complete.' "$TMP/harness.log"

echo "CACHE POLICY A/B HARNESS TEST PASSED"

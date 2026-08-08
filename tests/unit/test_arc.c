/* test_arc.c - checkpoint-free invariants for the ARC metadata engine. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_arc.h"

static int fail = 0;

static void ck(int ok, const char *what)
{
    printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fail++;
}

static int lru_hits(const int *tr, int n, int cap)
{
    int *keys = (int *)malloc((size_t)cap * sizeof(int));
    int *age = (int *)calloc((size_t)cap, sizeof(int));
    int used = 0, clock = 0, hits = 0;
    if (!keys || !age) { free(keys); free(age); return -1; }
    for (int i = 0; i < n; i++) {
        int slot = -1;
        for (int j = 0; j < used; j++) if (keys[j] == tr[i]) { slot = j; break; }
        if (slot >= 0) {
            hits++; age[slot] = ++clock; continue;
        }
        if (used < cap) slot = used++;
        else {
            slot = 0;
            for (int j = 1; j < used; j++) if (age[j] < age[slot]) slot = j;
        }
        keys[slot] = tr[i]; age[slot] = ++clock;
    }
    free(keys); free(age); return hits;
}

/* Small O(n*cap) Belady reference used only for randomized test traces. */
static int belady_hits(const int *tr, int n, int cap, int nkey)
{
    unsigned char *resident = (unsigned char *)calloc((size_t)nkey, 1);
    if (!resident) return -1;
    int nr = 0, hits = 0;
    for (int i = 0; i < n; i++) {
        const int key = tr[i];
        if (resident[key]) { hits++; continue; }
        if (nr >= cap) {
            int victim = -1, far = -1;
            for (int k = 0; k < nkey; k++) if (resident[k]) {
                int next = n;
                for (int j = i + 1; j < n; j++) if (tr[j] == k) { next = j; break; }
                if (next > far) { far = next; victim = k; }
            }
            if (victim >= 0) resident[victim] = 0;
            else { free(resident); return -1; }
        } else nr++;
        resident[key] = 1;
    }
    free(resident); return hits;
}

static int arc_hits(const int *tr, int n, int cap, int nkey)
{
    K3Arc *a = k3_arc_create(nkey, cap);
    unsigned char *resident = (unsigned char *)calloc((size_t)nkey, 1);
    int hits = 0;
    if (!a || !resident) { k3_arc_destroy(a); free(resident); return -1; }

    for (int i = 0; i < n; i++) {
        int hit = -1; int32_t victim = -2;
        const int key = tr[i];
        const int before = resident[key] != 0;
        if (k3_arc_access(a, key, &hit, &victim) != 0) {
            fail++; break;
        }
        if (hit != before) {
            fprintf(stderr, "ARC/physical residency mismatch at request %d key %d\n", i, key);
            fail++; break;
        }
        if (hit) {
            hits++;
            if (victim != -1) { fail++; break; }
        } else {
            if (victim >= 0) {
                if (victim >= nkey || !resident[victim]) {
                    fprintf(stderr, "ARC returned non-resident victim %d\n", (int)victim);
                    fail++; break;
                }
                resident[victim] = 0;
            }
            resident[key] = 1;
        }

        if (k3_arc_resident_count(a) > cap ||
            k3_arc_target(a) < 0.0 || k3_arc_target(a) > (double)cap) {
            fail++; break;
        }
        for (int k = 0; k < nkey; k++) {
            if (k3_arc_is_resident(a, k) != (resident[k] != 0)) {
                fprintf(stderr, "ARC membership mismatch at request %d key %d\n", i, k);
                fail++; i = n; break;
            }
        }
    }

    k3_arc_destroy(a); free(resident); return hits;
}

static uint32_t xs = 0x4b330001u;
static uint32_t xrnd(void)
{
    xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5; return xs;
}

int main(void)
{
    printf("ARC metadata policy\n\n");

    { int tr[20]; for (int i = 0; i < 20; i++) tr[i] = 7;
      ck(arc_hits(tr, 20, 1, 8) == 19, "one-slot repeated key"); }

    { int tr[100]; for (int i = 0; i < 100; i++) tr[i] = i;
      ck(arc_hits(tr, 100, 8, 100) == 0, "unique scan invents no hits"); }

    { int tr[24], n = 0;
      tr[n++] = 0; tr[n++] = 1; tr[n++] = 0; tr[n++] = 1;
      for (int k = 2; k < 20; k++) tr[n++] = k;
      tr[n++] = 0; tr[n++] = 1;
      const int ah = arc_hits(tr, n, 4, 20), lh = lru_hits(tr, n, 4);
      ck(ah > lh, "ARC resists a one-hit scan better than LRU"); }

    /* Randomized invariant: an online ARC may equal but cannot beat Belady. The helper
     * also checks after EVERY request that ARC's logical resident set exactly matches a
     * separately maintained physical set, which catches ghost/resident list corruption. */
    int random_ok = 1;
    for (int cap = 1; cap <= 8 && random_ok; cap++) {
        for (int rep = 0; rep < 40 && random_ok; rep++) {
            int tr[96];
            for (int i = 0; i < 96; i++) tr[i] = (int)(xrnd() % 16u);
            const int ah = arc_hits(tr, 96, cap, 16);
            const int bh = belady_hits(tr, 96, cap, 16);
            if (ah < 0 || bh < 0 || ah > bh) random_ok = 0;
        }
    }
    ck(random_ok, "random traces never exceed Belady and stay synchronized");

    printf("\n%s\n", fail ? "ARC TESTS FAILED" : "ARC TESTS PASSED");
    return fail ? 1 : 0;
}

/* k3_arc.h - internal Adaptive Replacement Cache metadata.
 *
 * This module owns NO expert bytes and performs NO I/O. It answers one question for a
 * logical cache of integer keys: on this access, was the key resident and, on a miss,
 * which resident key should be evicted? Keeping it separate makes the policy testable
 * without a checkpoint and keeps k3_cache.c responsible for physical slots only.
 */
#ifndef K3_ARC_H
#define K3_ARC_H

#include <stdint.h>

typedef struct K3Arc K3Arc;

K3Arc *k3_arc_create(int nkey, int capacity);
void   k3_arc_destroy(K3Arc *a);

/* Process one request. Returns 0 on success.
 *   *hit       1 if key was resident before this request, else 0
 *   *victim    resident key to evict on a miss, or -1 if a free slot is available
 */
int    k3_arc_access(K3Arc *a, int32_t key, int *hit, int32_t *victim);

/* Test/debug helpers. */
int    k3_arc_is_resident(const K3Arc *a, int32_t key);
int    k3_arc_resident_count(const K3Arc *a);
double k3_arc_target(const K3Arc *a);

#endif /* K3_ARC_H */

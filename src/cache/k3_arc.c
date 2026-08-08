/* k3_arc.c - Adaptive Replacement Cache metadata, no expert I/O. */
#include "k3_arc.h"

#include <stdlib.h>
#include <string.h>

/* List membership for each key. NONE is not on any ARC list. */
enum { ARC_NONE = 0, ARC_T1 = 1, ARC_T2 = 2, ARC_B1 = 3, ARC_B2 = 4 };

typedef struct {
    int32_t head, tail;
    int size;
} ArcList;

struct K3Arc {
    int nkey;
    int cap;
    double p;
    unsigned char *state;
    int32_t *prev;
    int32_t *next;
    ArcList list[5];
};

static ArcList *lst(K3Arc *a, int state) { return &a->list[state]; }
static const ArcList *clst(const K3Arc *a, int state) { return &a->list[state]; }

static void list_remove(K3Arc *a, int32_t key)
{
    const int s = a->state[key];
    if (s == ARC_NONE) return;
    ArcList *l = lst(a, s);
    const int32_t p = a->prev[key], n = a->next[key];
    if (p >= 0) a->next[p] = n; else l->head = n;
    if (n >= 0) a->prev[n] = p; else l->tail = p;
    a->prev[key] = a->next[key] = -1;
    a->state[key] = ARC_NONE;
    l->size--;
}

static void list_push_mru(K3Arc *a, int state, int32_t key)
{
    ArcList *l = lst(a, state);
    if (a->state[key] != ARC_NONE) list_remove(a, key);
    a->state[key] = (unsigned char)state;
    a->prev[key] = l->tail;
    a->next[key] = -1;
    if (l->tail >= 0) a->next[l->tail] = key; else l->head = key;
    l->tail = key;
    l->size++;
}

static int32_t list_pop_lru(K3Arc *a, int state)
{
    const int32_t key = lst(a, state)->head;
    if (key >= 0) list_remove(a, key);
    return key;
}

static void list_touch_mru(K3Arc *a, int state, int32_t key)
{
    if (lst(a, state)->tail == key) return;
    list_remove(a, key);
    list_push_mru(a, state, key);
}

/* ARC REPLACE. The incoming key is still in B1/B2 when called on a ghost hit, exactly as
 * in the published algorithm; that membership is what selects the tie-break at p. */
static int32_t arc_replace(K3Arc *a, int32_t incoming)
{
    ArcList *t1 = lst(a, ARC_T1), *t2 = lst(a, ARC_T2);
    const int incoming_b2 = a->state[incoming] == ARC_B2;
    const int use_t1 = t1->size > 0 &&
                       ((incoming_b2 && (double)t1->size >= a->p) ||
                        (double)t1->size > a->p);
    int32_t victim;
    if (use_t1 || t2->size == 0) {
        victim = list_pop_lru(a, ARC_T1);
        if (victim >= 0) list_push_mru(a, ARC_B1, victim);
    } else {
        victim = list_pop_lru(a, ARC_T2);
        if (victim >= 0) list_push_mru(a, ARC_B2, victim);
    }
    return victim;
}

K3Arc *k3_arc_create(int nkey, int capacity)
{
    if (nkey <= 0 || capacity <= 0) return NULL;
    K3Arc *a = (K3Arc *)calloc(1, sizeof *a);
    if (!a) return NULL;
    a->nkey = nkey;
    a->cap = capacity;
    for (int s = ARC_T1; s <= ARC_B2; s++)
        a->list[s].head = a->list[s].tail = -1;
    a->state = (unsigned char *)calloc((size_t)nkey, 1);
    a->prev = (int32_t *)malloc((size_t)nkey * sizeof(int32_t));
    a->next = (int32_t *)malloc((size_t)nkey * sizeof(int32_t));
    if (!a->state || !a->prev || !a->next) {
        k3_arc_destroy(a);
        return NULL;
    }
    for (int i = 0; i < nkey; i++) a->prev[i] = a->next[i] = -1;
    return a;
}

void k3_arc_destroy(K3Arc *a)
{
    if (!a) return;
    free(a->state);
    free(a->prev);
    free(a->next);
    free(a);
}

int k3_arc_access(K3Arc *a, int32_t key, int *hit, int32_t *victim)
{
    if (!a || key < 0 || key >= a->nkey || !hit || !victim) return -1;
    *hit = 0;
    *victim = -1;

    const int s = a->state[key];
    if (s == ARC_T1) {
        *hit = 1;
        list_remove(a, key);
        list_push_mru(a, ARC_T2, key);
        return 0;
    }
    if (s == ARC_T2) {
        *hit = 1;
        list_touch_mru(a, ARC_T2, key);
        return 0;
    }

    if (s == ARC_B1) {
        const int b1 = clst(a, ARC_B1)->size, b2 = clst(a, ARC_B2)->size;
        const double delta = b1 >= b2 ? 1.0 : (double)b2 / (double)(b1 ? b1 : 1);
        a->p += delta;
        if (a->p > a->cap) a->p = a->cap;
        *victim = arc_replace(a, key);
        /* REPLACE may grow B1 enough to evict the incoming ghost only in a broken
         * implementation. list_remove is intentionally idempotent so metadata remains
         * safe even under a boundary tie. */
        list_remove(a, key);
        list_push_mru(a, ARC_T2, key);
        return 0;
    }

    if (s == ARC_B2) {
        const int b1 = clst(a, ARC_B1)->size, b2 = clst(a, ARC_B2)->size;
        const double delta = b2 >= b1 ? 1.0 : (double)b1 / (double)(b2 ? b2 : 1);
        a->p -= delta;
        if (a->p < 0.0) a->p = 0.0;
        *victim = arc_replace(a, key);
        list_remove(a, key);
        list_push_mru(a, ARC_T2, key);
        return 0;
    }

    ArcList *t1 = lst(a, ARC_T1), *t2 = lst(a, ARC_T2);
    ArcList *b1 = lst(a, ARC_B1), *b2 = lst(a, ARC_B2);

    if (t1->size + b1->size == a->cap) {
        if (t1->size < a->cap) {
            (void)list_pop_lru(a, ARC_B1);
            *victim = arc_replace(a, key);
        } else {
            /* Standard ARC case IV.A: T1 itself fills the cache, so evict its LRU
             * directly rather than retaining a ghost and exceeding the metadata bound. */
            *victim = list_pop_lru(a, ARC_T1);
        }
    } else if (t1->size + b1->size < a->cap) {
        const int total = t1->size + t2->size + b1->size + b2->size;
        if (total >= a->cap) {
            if (total >= 2 * a->cap) (void)list_pop_lru(a, ARC_B2);
            *victim = arc_replace(a, key);
        }
    }

    list_push_mru(a, ARC_T1, key);
    return 0;
}

int k3_arc_is_resident(const K3Arc *a, int32_t key)
{
    if (!a || key < 0 || key >= a->nkey) return 0;
    return a->state[key] == ARC_T1 || a->state[key] == ARC_T2;
}

int k3_arc_resident_count(const K3Arc *a)
{
    if (!a) return 0;
    return clst(a, ARC_T1)->size + clst(a, ARC_T2)->size;
}

double k3_arc_target(const K3Arc *a)
{
    return a ? a->p : 0.0;
}

/* test_trunk.c - weightless regression coverage for the trunk streaming ring.
 *
 * The asynchronous reader needs two ring slots: one for the layer currently being
 * consumed and one for the next layer being read. A one-slot ring used to start the
 * reader anyway, allowing prefetch(L+1) to overwrite L while the caller still held
 * pointers into it. The run completed normally with different tokens.
 *
 * This test builds a tiny, structurally valid two-layer packed trunk at runtime. Each
 * 4 KiB layer has a distinct byte pattern and a complete one-element KDA+dense tensor
 * set, so k3_trunk_bind() follows the real binder path without model weights.
 */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "k3_trunk.h"

#define RUN_BYTES 4096

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: ");                                        \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                           \
            goto fail;                                                         \
        }                                                                      \
    } while (0)

typedef struct {
    const char *suffix;
    const char *dtype;
    int nbytes;
} TensorSpec;

/* With every configured dimension equal to one, every tensor expected by a KDA+dense
 * layer contains exactly one element. Large matrices stay BF16; vectors consumed
 * elementwise are F32, matching k3_bind_layer_mem's storage contract. */
static const TensorSpec SPECS[] = {
    { "input_layernorm.weight",                 "F32",  4 },
    { "post_attention_layernorm.weight",        "F32",  4 },
    { "self_attention_res_norm.weight",         "F32",  4 },
    { "self_attention_res_proj.weight",         "F32",  4 },
    { "mlp_res_norm.weight",                    "F32",  4 },
    { "mlp_res_proj.weight",                    "F32",  4 },

    { "self_attn.q_proj.weight",                "BF16", 2 },
    { "self_attn.k_proj.weight",                "BF16", 2 },
    { "self_attn.v_proj.weight",                "BF16", 2 },
    { "self_attn.g_proj.weight",                "BF16", 2 },
    { "self_attn.o_proj.weight",                "BF16", 2 },
    { "self_attn.q_conv1d.weight",              "F32",  4 },
    { "self_attn.k_conv1d.weight",              "F32",  4 },
    { "self_attn.v_conv1d.weight",              "F32",  4 },
    { "self_attn.f_a_proj.weight",              "BF16", 2 },
    { "self_attn.f_b_proj.weight",              "BF16", 2 },
    { "self_attn.b_proj.weight",                "BF16", 2 },
    { "self_attn.A_log",                        "F32",  4 },
    { "self_attn.dt_bias",                      "F32",  4 },
    { "self_attn.o_norm.weight",                "F32",  4 },

    { "mlp.gate_proj.weight",                   "BF16", 2 },
    { "mlp.up_proj.weight",                     "BF16", 2 },
    { "mlp.down_proj.weight",                   "BF16", 2 },
};

static K3Cfg tiny_cfg(void)
{
    K3Cfg c;
    memset(&c, 0, sizeof c);
    c.hidden = 1;
    c.n_layers = 2;
    c.vocab = 1;
    c.rms_eps = 1e-5f;

    c.kda_heads = 1;
    c.kda_head_dim = 1;
    c.conv_k = 1;
    c.gate_lb = -5.0f;

    /* MLA is not selected in this fixture, but these values participate in the shared
     * widen-area sizing and therefore stay valid rather than zero by accident. */
    c.n_heads = 1;
    c.q_lora = 1;
    c.kv_lora = 1;
    c.qk_nope = 1;
    c.qk_rope = 0;
    c.v_head = 1;
    c.mla_out_gate = 1;

    c.n_experts = 1;
    c.topk = 1;
    c.n_shared = 1;
    c.latent = 1;
    c.moe_inter = 1;
    c.routed_scale = 1.0f;
    c.moe_renorm = 1;
    c.latent_norm = 1;

    /* Both layers are dense so the fixture needs no routed-expert trunk tensors. */
    c.first_dense = 2;
    c.dense_inter = 1;
    c.attn_res_block = 1;
    c.situ_b1 = 4.0f;
    c.situ_b2 = 25.0f;
    c.n_full_attn = 0;
    c.full_attn = NULL;
    return c;
}

static int64_t ring_slot_bytes(const K3Cfg *c)
{
    int64_t n = RUN_BYTES;
    n = (n + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);
    n += (int64_t)k3_bind_widen_bytes(c);
    n = (n + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);
    return n;
}

static int write_layer_json(FILE *f, int layer, int64_t file_off)
{
    if (fprintf(f, "{\"file_off\":%lld,\"nbytes\":%d,\"tensors\":{",
                (long long)file_off, RUN_BYTES) < 0)
        return -1;

    for (size_t i = 0; i < sizeof SPECS / sizeof SPECS[0]; i++) {
        const int off = (int)i * 8; /* aligned for both F32 and BF16 */
        if (fprintf(f,
                    "%s\"language_model.model.layers.%d.%s\":"
                    "{\"off\":%d,\"nbytes\":%d,\"dtype\":\"%s\"}",
                    i ? "," : "", layer, SPECS[i].suffix,
                    off, SPECS[i].nbytes, SPECS[i].dtype) < 0)
            return -1;
    }
    return fprintf(f, "}}") < 0 ? -1 : 0;
}

static int make_fixture(char *dir, size_t cap)
{
    snprintf(dir, cap, "/tmp/k3-trunk-test-%ld", (long)getpid());
    if (mkdir(dir, 0700) != 0) {
        perror("mkdir trunk fixture");
        return -1;
    }

    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    FILE *bin = fopen(path, "wb");
    if (!bin) return -1;
    unsigned char run[RUN_BYTES];
    memset(run, 0x11, sizeof run);
    if (fwrite(run, 1, sizeof run, bin) != sizeof run) { fclose(bin); return -1; }
    memset(run, 0x22, sizeof run);
    if (fwrite(run, 1, sizeof run, bin) != sizeof run) { fclose(bin); return -1; }
    if (fclose(bin) != 0) return -1;

    snprintf(path, sizeof path, "%s/trunk.json", dir);
    FILE *json = fopen(path, "wb");
    if (!json) return -1;
    int rc = 0;
    if (fprintf(json, "{\"align\":%d,\"layers\":[", K3_TRUNK_ALIGN) < 0) rc = -1;
    if (!rc && write_layer_json(json, 0, 0) != 0) rc = -1;
    if (!rc && fprintf(json, ",") < 0) rc = -1;
    if (!rc && write_layer_json(json, 1, RUN_BYTES) != 0) rc = -1;
    if (!rc && fprintf(json, "]}\n") < 0) rc = -1;
    if (fclose(json) != 0) rc = -1;
    return rc;
}

static void remove_fixture(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/trunk.bin", dir); unlink(path);
    snprintf(path, sizeof path, "%s/trunk.json", dir); unlink(path);
    rmdir(dir);
}

static int find_layer_slot(const K3Trunk *tr, int layer)
{
    for (int i = 0; i < tr->nslot; i++)
        if (tr->layer_of[i] == layer) return i;
    return -1;
}

static int test_one_slot(const char *dir, const K3Cfg *cfg, int64_t slot_bytes)
{
    K3Trunk tr;
    K3LayerBind bind;
    memset(&tr, 0, sizeof tr);
    memset(&bind, 0, sizeof bind);

    CHECK(k3_trunk_open(&tr, dir, cfg, slot_bytes) == 0, "one-slot open failed");
    CHECK(tr.npin == 0, "one-slot fixture unexpectedly pinned %d layers", tr.npin);
    CHECK(tr.nslot == 1, "budget should produce one ring slot, got %d", tr.nslot);
    CHECK(tr.io_state == NULL,
          "one-slot ring started the async reader; prefetch could overwrite live data");

    CHECK(k3_trunk_bind(&tr, cfg, 0, &bind) == 0, "binding layer 0 failed");
    CHECK(tr.layer_of[0] == 0, "layer 0 is not resident in the only slot");
    CHECK(tr.arena[RUN_BYTES - 1] == 0x11, "layer 0 payload was not loaded intact");

    /* With no reader, this must be a complete no-op. The following bind then performs
     * the layer-1 read synchronously into the same slot. */
    k3_trunk_prefetch(&tr, 1);
    CHECK(tr.layer_of[0] == 0, "one-slot prefetch evicted the layer being consumed");
    CHECK(tr.arena[RUN_BYTES - 1] == 0x11, "one-slot prefetch overwrote live layer bytes");

    CHECK(k3_trunk_bind(&tr, cfg, 1, &bind) == 0, "binding layer 1 failed");
    CHECK(tr.layer_of[0] == 1, "layer 1 did not replace layer 0 synchronously");
    CHECK(tr.arena[RUN_BYTES - 1] == 0x22, "layer 1 payload was not loaded intact");

    k3_trunk_close(&tr);
    return 0;

fail:
    k3_trunk_close(&tr);
    return 1;
}

static int test_two_slots(const char *dir, const K3Cfg *cfg, int64_t slot_bytes)
{
    K3Trunk tr;
    K3LayerBind bind;
    memset(&tr, 0, sizeof tr);
    memset(&bind, 0, sizeof bind);

    CHECK(k3_trunk_open(&tr, dir, cfg, 2 * slot_bytes) == 0, "two-slot open failed");
    CHECK(tr.npin == 0, "two-slot fixture unexpectedly pinned %d layers", tr.npin);
    CHECK(tr.nslot == 2, "budget should produce two ring slots, got %d", tr.nslot);
    CHECK(tr.io_state != NULL, "two-slot ring did not start the async reader");

    CHECK(k3_trunk_bind(&tr, cfg, 0, &bind) == 0, "binding layer 0 failed");
    const int slot0 = find_layer_slot(&tr, 0);
    CHECK(slot0 >= 0, "layer 0 is not resident after bind");
    CHECK(tr.arena[(size_t)slot0 * tr.slot_bytes + RUN_BYTES - 1] == 0x11,
          "layer 0 payload was not loaded intact");

    k3_trunk_prefetch(&tr, 1);
    /* Binding layer 1 waits for the in-flight read, so this is deterministic and needs
     * no sleeps or racy polling of the reader-owned state. */
    CHECK(k3_trunk_bind(&tr, cfg, 1, &bind) == 0, "binding prefetched layer 1 failed");
    const int slot1 = find_layer_slot(&tr, 1);
    CHECK(slot1 >= 0, "prefetched layer 1 was not published resident");
    CHECK(slot1 != slot0, "prefetch reused the live layer-0 slot");
    CHECK(tr.layer_of[slot0] == 0, "prefetch evicted layer 0 before its slot was reusable");
    CHECK(tr.arena[(size_t)slot0 * tr.slot_bytes + RUN_BYTES - 1] == 0x11,
          "async prefetch overwrote the layer-0 payload");
    CHECK(tr.arena[(size_t)slot1 * tr.slot_bytes + RUN_BYTES - 1] == 0x22,
          "async prefetch did not load the layer-1 payload intact");

    k3_trunk_close(&tr);
    return 0;

fail:
    k3_trunk_close(&tr);
    return 1;
}

int main(void)
{
    char dir[256];
    if (make_fixture(dir, sizeof dir) != 0) {
        fprintf(stderr, "FAIL: could not create synthetic trunk fixture\n");
        remove_fixture(dir);
        return 1;
    }

    const K3Cfg cfg = tiny_cfg();
    const int64_t slot = ring_slot_bytes(&cfg);
    int rc = 0;

    printf("trunk async regression: synthetic slot = %.1f KiB\n", (double)slot / 1024.0);
    if (test_one_slot(dir, &cfg, slot) != 0) rc = 1;
    if (!rc && test_two_slots(dir, &cfg, slot) != 0) rc = 1;

    remove_fixture(dir);
    if (rc) return 1;
    printf("TRUNK ASYNC TEST PASSED\n");
    return 0;
}

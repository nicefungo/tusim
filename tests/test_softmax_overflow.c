/*
 * TU CModel — Softmax overflow-admission gate
 * ============================================
 * Gate for the 2026-11-13 fix: uint32 endpoint arithmetic in
 * tu_softmax_validate_desc (data_offset/out_offset + elem_count*4) and
 * row*column products in the 2D convenience wrappers wrapped before
 * comparison, admitting fictitious work as success (SRAM pre-admission
 * silently rejected the accesses; the engine reported stall cycles
 * from uninitialized rows).
 *
 * Alternatives considered:
 *   - checked uint64 admission (implemented): rejects before any access,
 *     no partial writes, constant host cost.
 *   - wider/segmented execution (BLOCKED): needs new extent/scheduling and
 *     error contracts; not a validation fix.
 *   - unchecked (EXCLUDED): unsafe; silent prefix normalization of huge
 *     tensors and garbage output from OOB-rejected reads.
 *
 * Valid controls prove correct shapes are unaffected.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../tu_cmodel/tu_config.h"
#include "../tu_cmodel/tu_sram.h"
#include "../tu_cmodel/compute/softmax_engine.h"

static int failures = 0;

static void expect_reject(const char *name, uint64_t got) {
    if (got == UINT64_MAX) {
        printf("PASS %-28s rejected\n", name);
    } else {
        printf("FAIL %-28s returned %llu (false success)\n",
               name, (unsigned long long)got);
        failures++;
    }
}

static void expect_ok(const char *name, uint64_t got) {
    if (got != UINT64_MAX) {
        printf("PASS %-28s accepted (%llu stalls)\n",
               name, (unsigned long long)got);
    } else {
        printf("FAIL %-28s rejected valid shape\n", name);
        failures++;
    }
}

int main(void) {
    /* --- Invalid: descriptor data-end wrap --- */
    {
        tu_sram_region_t r;
        static uint8_t backing[1024];
        memset(&r, 0, sizeof(r));
        r.total_size = sizeof(backing);
        r.banks.data = backing;

        tu_softmax_desc_t d;
        memset(&d, 0, sizeof(d));
        d.mode = TU_SOFTMAX_STANDARD;
        d.data_sram = &r;
        d.data_offset = 0xFFFFFFF8u;
        d.elem_count = 2;
        d.axis_dim = 0;
        d.in_place = true;
        expect_reject("data-offset wrap", tu_softmax_execute(&d));
    }

    /* --- Invalid: descriptor out-end wrap (out-of-place) --- */
    {
        tu_sram_region_t r;
        static uint8_t backing[1024];
        memset(&r, 0, sizeof(r));
        r.total_size = sizeof(backing);
        r.banks.data = backing;

        tu_softmax_desc_t d;
        memset(&d, 0, sizeof(d));
        d.mode = TU_SOFTMAX_STANDARD;
        d.data_sram = &r;
        d.data_offset = 0;
        d.elem_count = 1;
        d.axis_dim = 0;
        d.in_place = false;
        d.out_offset = 0xFFFFFFFCu;
        expect_reject("out-offset wrap", tu_softmax_execute(&d));
    }

    /* --- Invalid: elem_count whose byte extent itself overflows uint32 --- */
    {
        tu_sram_region_t r;
        static uint8_t backing[1024];
        memset(&r, 0, sizeof(r));
        r.total_size = sizeof(backing);
        r.banks.data = backing;

        tu_softmax_desc_t d;
        memset(&d, 0, sizeof(d));
        d.mode = TU_SOFTMAX_STANDARD;
        d.data_sram = &r;
        d.data_offset = 0;
        d.elem_count = 0x40000000u;   /* *4 = 2^32, wraps to 0 unfixed */
        d.axis_dim = 1;
        d.in_place = true;
        expect_reject("elem_count byte-extent wrap", tu_softmax_execute(&d));
    }

    /* --- Invalid: 2D wrapper row*column product wraps to a fitting prefix --- */
    {
        tu_sram_region_t r;
        tu_sram_init(&r, 0x10000u * 4u, "smx-wrap");

        /* rows*cols = 0x10000 * 0x10001 = 0x1_0001_0000 -> wraps to 0x10000,
         * which is divisible by cols and fits the region */
        expect_reject("wrapper rows*cols wrap",
                      tu_softmax_2d(&r, 0, 0x10001u, 0x10000u, 0.0f, true));
        expect_reject("masked wrapper wrap",
                      tu_softmax_masked(&r, 0, 0x10001u, 0x10000u,
                                        NULL, -1e9f, 0.0f));
        expect_reject("log wrapper wrap",
                      tu_log_softmax_2d(&r, 0, 0x10001u, 0x10000u,
                                        0.0f, true));
        tu_sram_destroy(&r);
    }

    /* --- Valid controls: shapes must still work and be numerically sane --- */
    {
        tu_sram_region_t r;
        tu_sram_init(&r, 16 * 1024, "smx-ok");

        /* 4 rows x 8 cols distinct values */
        float vals[32];
        for (uint32_t i = 0; i < 32; i++) vals[i] = (float)((int)i - 16);
        uint8_t *backing = r.banks.data;
        memcpy(backing, vals, sizeof(vals));

        uint64_t s = tu_softmax_2d(&r, 0, 4, 8, 0.0f, true);
        expect_ok("valid 4x8 in-place", s);
        float out[32];
        memcpy(out, backing, sizeof(out));
        double row_sum = 0.0;
        for (uint32_t j = 0; j < 8; j++) row_sum += out[j];
        if (fabs(row_sum - 1.0) > 1e-5 || !(out[7] > out[0])) {
            printf("FAIL valid control numerics (row sum %f)\n", row_sum);
            failures++;
        } else {
            printf("PASS valid control numerics\n");
        }

        /* out-of-place valid path (exercises the fixed out-end printf) */
        expect_ok("valid out-of-place",
                  tu_softmax(&r, 0, 8, 0.0f, false));

        /* exact-fit boundary: last valid element ends exactly at region end */
        uint32_t n = r.total_size / sizeof(float);   /* fills region */
        memcpy(backing, vals, sizeof(vals));
        expect_ok("exact-fit full region",
                  tu_softmax(&r, 0, n, 0.0f, true));

        /* one element past the end must still reject (non-wrapping path) */
        expect_reject("plain overflow (no wrap)",
                      tu_softmax(&r, 0, n + 1, 0.0f, true));

        tu_sram_destroy(&r);
    }

    printf("test-softmax-overflow: %s (%d failures)\n",
           failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

/* Pinned observation probe, NOT a normalization conformance test. */
#include "tu_cmodel/compute/normalization_engine.h"
#include "tu_cmodel/compute/softmax_engine.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    tu_sram_region_t s;
    tu_sram_init(&s, 128, "probe");
    int failures = 0;
    const float input[5] = {1, 3, 5, 7, 9};
    printf("engine,mode,count,axis,status,last_output,tail_unchanged,stats_row2_unchanged\n");
    for (int mode = 0; mode < 2; ++mode) {
        for (unsigned count = 4; count <= 5; ++count) {
            float sentinel[5] = {99, 99, 99, 99, 99};
            float mean[3] = {777,777,777}, var[3] = {777,777,777};
            memcpy(tu_sram_raw_ptr(&s), input, sizeof input);
            memcpy((unsigned char *)tu_sram_raw_ptr(&s) + 64, sentinel, sizeof sentinel);
            tu_norm_desc_t d = {.mode = mode, .data_sram = &s, .elem_count = count,
                .epsilon = 1e-5f, .norm_axis_dim = 2, .in_place = false,
                .out_offset = 64, .mean_out = mode == 0 ? mean : NULL, .var_out = var};
            uint64_t status = tu_norm_execute(&d);
            float out[5]; memcpy(out, (unsigned char *)tu_sram_raw_ptr(&s) + 64, sizeof out);
            int tail = out[4] == 99.0f;
            int stat = var[2] == 777.0f;
            printf("norm,%s,%u,2,%s,%.6g,%d,%d\n", mode ? "rms" : "layer", count,
                   status == UINT64_MAX ? "error" : "success", out[4], tail, stat);
            if (status == UINT64_MAX || !tail || !stat || out[0] == 99.0f || out[3] == 99.0f) failures++;
        }
    }
    for (unsigned count = 4; count <= 5; ++count) {
        memcpy(tu_sram_raw_ptr(&s), input, sizeof input);
        tu_softmax_desc_t d = {.mode = TU_SOFTMAX_STANDARD, .data_sram = &s,
            .elem_count = count, .axis_dim = 2, .in_place = true};
        int valid = tu_softmax_validate_desc(&d);
        printf("softmax,standard,%u,2,%s,-,-,-\n", count, valid ? "valid" : "invalid");
        if (valid != (count == 4)) failures++;
    }
    tu_norm_desc_t d = {.mode = TU_NORM_LAYER_NORM, .data_sram = &s,
        .elem_count = 2, .epsilon = 1e-5f, .in_place = false, .out_offset = 120};
    printf("validation_case,accepted\n");
    printf("exact_end,%d\n", tu_norm_validate_desc(&d));
    if (!tu_norm_validate_desc(&d)) failures++;
    d.out_offset = 124;
    printf("out_of_bounds_output,%d\n", tu_norm_validate_desc(&d));
    d.out_offset = 0; d.data_offset = UINT32_MAX - 3;
    printf("wrapped_input,%d\n", tu_norm_validate_desc(&d));
    d.data_offset = 0; d.gamma_sram = &s; d.gamma_offset = 124;
    printf("out_of_bounds_gamma,%d\n", tu_norm_validate_desc(&d));
    d.gamma_sram = NULL; d.beta_sram = &s; d.beta_offset = 124;
    printf("out_of_bounds_beta,%d\n", tu_norm_validate_desc(&d));
    tu_sram_destroy(&s);
    printf("snapshot_failures,%d\n", failures);
    return failures ? 1 : 0;
}

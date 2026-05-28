/*
 * TU ASM Interpreter Test — loads .tuasm file, executes, verifies output.
 */
#include "tu_cmodel/tu_cmodel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    /* Load ASM program from file */
    FILE *af = fopen("/tmp/single_linear.tuasm", "rb");
    if (!af) { fprintf(stderr, "Cannot open /tmp/single_linear.tuasm\n"); return 1; }
    fseek(af, 0, SEEK_END);
    long sz = ftell(af);
    fseek(af, 0, SEEK_SET);
    char *asm_prog = malloc(sz + 1);
    if (!asm_prog) return 1;
    fread(asm_prog, 1, sz, af);
    asm_prog[sz] = '\0';
    fclose(af);

    /* Load input (pre-transposed for ASM: X^T[64][2]) */
    fp16_t input_data[128];
    FILE *f = fopen("/tmp/tu_input_asm.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open input\n"); return 1; }
    fread(input_data, 1, 256, f);
    fclose(f);

    /* Output buffer */
    fp32_t output_data[64];

    tu_host_buffer_t bufs[] = {
        {"input",  input_data,  256},
        {"output", output_data, 256},
    };

    printf("=== TU ASM Interpreter ===\n");
    int rc = tu_run_asm(asm_prog, bufs, 2);
    free(asm_prog);

    if (rc != 0) { fprintf(stderr, "ASM execution failed\n"); return 1; }
    tu_print_stats();

    printf("=== ASM OUTPUT (O[N][M], N=32 M=2) ===\n");
    for (int n = 0; n < 5; n++)
        printf("  row %2d: [ %12.8f  %12.8f ]\n",
               n, output_data[n*2], output_data[n*2+1]);

    /* Compare with C API output */
    fp32_t expected[64];
    f = fopen("/tmp/tu_output.bin", "rb");
    if (f) {
        fread(expected, sizeof(fp32_t), 64, f);
        fclose(f);
        float max_err = 0.0f;
        for (int i = 0; i < 64; i++) {
            float err = fabsf(output_data[i] - expected[i]);
            if (err > max_err) max_err = err;
        }
        printf("\nMax error vs C API output: %.10f\n", max_err);
        printf("Match: %s\n", max_err < 0.000001f ? "YES (identical)" : "CLOSE");
    }

    return 0;
}

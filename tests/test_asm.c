/*
 * TU ASM Interpreter Smoke Test
 * =============================
 *
 * Self-contained regression: execute a 2x2 identity MMA from an in-memory
 * ASM program.  The old test depended on three undocumented /tmp artifacts,
 * so a clean checkout could never run `make test-quick`.
 */
#include "tu_cmodel/tu_cmodel.h"

#include <math.h>
#include <stdio.h>

int main(void)
{
    /* FP16 little-endian W = [[1, 0], [0, 1]]. */
    static const char program[] =
        "%weight identity\n"
        "00 3C 00 00 00 00 00 3C\n"
        "%endweight\n"
        "%input input 8\n"
        "%output output 16\n"
        "LOAD_W 0, 8\n"
        "LOAD_A input, 0, 8\n"
        "MMA 2, 2, 2, 0, 0, 0, NOBIAS\n"
        "STORE_O output, 0, 16\n"
        "SYNC\n";

    fp16_t input[4] = {
        tu_fp32_to_fp16(1.0f), tu_fp32_to_fp16(2.0f),
        tu_fp32_to_fp16(3.0f), tu_fp32_to_fp16(4.0f),
    };
    fp32_t output[4] = {0};
    const tu_host_buffer_t buffers[] = {
        {"input", input, sizeof(input)},
        {"output", output, sizeof(output)},
    };

    if (tu_run_asm(program, buffers, 2) != 0) {
        fprintf(stderr, "ASM execution failed\n");
        return 1;
    }

    for (int i = 0; i < 4; i++) {
        const float expected = (float)(i + 1);
        if (fabsf(output[i] - expected) > 1e-6f) {
            fprintf(stderr, "output[%d]=%.8f, expected %.8f\n",
                    i, output[i], expected);
            return 1;
        }
    }

    printf("TU ASM self-contained identity smoke test: PASS\n");
    return 0;
}

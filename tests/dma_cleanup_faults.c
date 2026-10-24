/* Test-only link wrappers: exercise DMA harness failure cleanup, not hardware.
 * No production allocation, submission, ownership or timing policy is changed. */
#include "tu_cmodel/dma_descriptor.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned allocations, submissions, ticks;

static unsigned setting(const char *name) {
    const char *s = getenv(name);
    return s ? (unsigned)strtoul(s, NULL, 10) : 0;
}

void *__real_calloc(size_t count, size_t size);
void *__wrap_calloc(size_t count, size_t size) {
    /* Scope is descriptor objects only, not SRAM or multicast metadata OOM. */
    if (count == 1 && size == sizeof(tu_dma_descriptor_t)) {
        ++allocations;
        if (allocations == setting("DMA_CLEANUP_FAIL_ALLOC")) {
            fprintf(stderr, "INJECT allocation %u\n", allocations);
            return NULL;
        }
    }
    return __real_calloc(count, size);
}

uint32_t __real_tu_dma_submit_desc(tu_dma_descriptor_t *desc);
uint32_t __wrap_tu_dma_submit_desc(tu_dma_descriptor_t *desc) {
    ++submissions;
    if (desc && submissions == setting("DMA_CLEANUP_REJECT_SUBMIT")) {
        fprintf(stderr, "INJECT rejection %u\n", submissions);
        /* Real span admission rejects unsupported directions and destroys
         * the entire chain. Do not synthesize a successful or failed ID. */
        desc->direction = (tu_dma_direction_t)255;
    }
    static bool early_injected;
    uint32_t id = __real_tu_dma_submit_desc(desc);
    if (id && g_tu_dma.async_mode && !early_injected && setting("DMA_CLEANUP_EARLY_DATA")) {
        early_injected = true;
        if (!desc->dst_region || !desc->total_bytes) abort();
        fprintf(stderr, "INJECT pre-tick data %u\n", submissions);
        ((unsigned char *)tu_sram_raw_ptr(desc->dst_region))[desc->dst_base] = 0x7e;
    }
    return id;
}

int __real_tu_dma_tick(void);
int __wrap_tu_dma_tick(void) {
    ++ticks;
    if (setting("DMA_CLEANUP_HOLD_ACTIVE") && ticks > 1) return 0;
    if (setting("DMA_CLEANUP_HOLD_ACTIVE") && ticks == 1)
        fprintf(stderr, "INJECT active hold\n");
    return __real_tu_dma_tick();
}

__attribute__((destructor)) static void report_counts(void) {
    fprintf(stderr, "CLEANUP_COUNTS allocations=%u submissions=%u ticks=%u\n",
            allocations, submissions, ticks);
}

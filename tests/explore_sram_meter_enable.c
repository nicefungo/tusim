/* Executable observation of SRAM meter disable/re-enable, not a physical timing model. */
#include "tu_cmodel/tu_sram.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const uint64_t gaps[] = {0, 3, 4, 8};
    unsigned cases = 0, failures = 0;
    puts("direction,disabled,gap,first,after_advance0,served,stalled,global_words,last_refill,data_ok");
    for (unsigned write = 0; write < 2; ++write) {
        for (unsigned disabled = 0; disabled < 2; ++disabled) {
            for (unsigned k = 0; k < sizeof gaps / sizeof gaps[0]; ++k) {
                tu_sram_region_t r;
                uint32_t input = 0x12345678u, output = 0;
                tu_sram_init_runtime(&r, 64, "enable-probe", 1, 4, 1, 2, 4);
                memcpy(r.banks.data, &input, 4);
                uint64_t s0 = write ? tu_sram_write(&r, 0, &input) : tu_sram_read(&r, 0, &output);
                if (disabled) tu_sram_set_bw_modeling(&r, false);
                tu_sram_advance_cycle(&r, gaps[k]);
                /* Access once only in the disabled case; this must copy but not meter. */
                if (disabled) {
                    uint32_t disabled_value = 0xa5c37e19u;
                    uint64_t s = write ? tu_sram_write(&r, 0, &disabled_value) : tu_sram_read(&r, 0, &output);
                    uint32_t copied; memcpy(&copied, r.banks.data, 4);
                    if (s != 0 || (write ? copied != disabled_value : output != input) ||
                        r.banks.bw_banks[0].words_available != 0 ||
                        r.banks.bw_banks[0].reads_served + r.banks.bw_banks[0].writes_served != 1 ||
                        r.banks.bw_banks[0].read_stalls + r.banks.bw_banks[0].write_stalls != 0 ||
                        r.banks.bw_banks[0].last_refill_cycle != 0) ++failures;
                    tu_sram_set_bw_modeling(&r, true);
                }
                uint64_t s1 = write ? tu_sram_write(&r, 0, &input) : tu_sram_read(&r, 0, &output);
                tu_sram_advance_cycle(&r, 0);
                uint64_t s2 = write ? tu_sram_write(&r, 0, &input) : tu_sram_read(&r, 0, &output);
                const tu_sram_bw_bank_t *b = &r.banks.bw_banks[0];
                uint64_t served = b->reads_served + b->writes_served;
                uint64_t stalled = b->read_stalls + b->write_stalls;
                uint64_t words = write ? r.banks.writes : r.banks.reads;
                uint32_t observed; memcpy(&observed, r.banks.data, 4);
                int data_ok = observed == input && (write || output == input);
                printf("%s,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
                       write ? "write" : "read", disabled, gaps[k], s1, s2, served, stalled, words, b->last_refill_cycle, data_ok);
                uint64_t e1 = (!disabled && gaps[k] >= 4) ? 0 : 2;
                uint64_t e2 = (disabled && gaps[k] >= 4) ? 0 : 2;
                uint64_t es = 1 + (e1 == 0) + (e2 == 0);
                if (s0 || s1 != e1 || s2 != e2 || served != es || stalled != 3-es ||
                    words != 3+disabled || b->last_refill_cycle != (gaps[k] >= 4 ? gaps[k] : 0) || !data_ok) ++failures;
                tu_sram_destroy(&r);
                ++cases;
            }
        }
    }
    fprintf(stderr, "cases=%u failures=%u\n", cases, failures);
    return failures ? 1 : 0;
}

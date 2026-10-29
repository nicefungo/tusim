/* Diagnostic, not a safety regression: the backing allocation is intentionally
 * larger than the advertised region, so invalid logical accesses are observable
 * without a host out-of-bounds access. Do not generalize to arbitrary inputs. */
#include "tu_cmodel/tu_sram.h"
#include "tu_cmodel/tu_status.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int probe(const char *name, int bulk, int write, uint32_t offset) {
    tu_sram_region_t r;
    uint8_t input[4] = {1, 2, 3, 4}, output[4] = {0};
    tu_sram_init_runtime(&r, 32, "diagnostic", 4, 4, 1, 2, 4);
    if (!r.banks.data || !r.banks.bw_banks) return 1;
    memset(r.banks.data, 0x5a, 32);
    r.total_size = 16; /* logical capacity; keep physical host allocation 32 */
    tu_clear_error();
    uint64_t stall;
    if (write) {
        stall = bulk ? tu_sram_write_bulk(&r, offset, input, 4)
                     : tu_sram_write(&r, offset, input);
    } else {
        stall = bulk ? tu_sram_read_bulk(&r, offset, output, 4)
                     : tu_sram_read(&r, offset, output);
    }
    int moved = write ? memcmp(r.banks.data + offset, input, 4) == 0
                      : output[0] == 0x5a && output[1] == 0x5a &&
                        output[2] == 0x5a && output[3] == 0x5a;
    const tu_error_t *last_error = tu_get_last_error();
    int error = last_error ? (int)last_error->code : TU_OK;
    printf("%-12s %-5s offset=%2" PRIu32 " valid=%d moved=%d error=%d reads=%" PRIu64 " writes=%" PRIu64 " stall=%" PRIu64 "\n",
           name, write ? "write" : "read", offset, offset <= 12, moved,
           error, r.banks.reads, r.banks.writes, stall);
    int failed = offset == 12 && (!moved || error != TU_OK || stall != 0 ||
                                 (write ? r.banks.writes != 1 : r.banks.reads != 1));
    tu_sram_destroy(&r);
    return failed;
}

int main(void) {
    tu_set_error_mode(TU_ERR_MODE_SILENT);
    int failed = 0;
    for (int bulk = 0; bulk < 2; bulk++)
        for (int write = 0; write < 2; write++)
            for (uint32_t offset = 12; offset <= 16; offset += 2)
                failed |= probe(bulk ? "bulk" : "scalar", bulk, write, offset);
    uint32_t addr = UINT32_MAX - 1u, size = 4, capacity = 16;
    uint32_t wrapped = addr + size;
    printf("arithmetic   addr=%" PRIu32 " size=%" PRIu32 " wrapped=%" PRIu32
           " legacy_reject=%d wide_reject=%d (not dereferenced)\n",
           addr, size, wrapped, wrapped > capacity,
           (uint64_t)addr + size > capacity);
    return failed;
}

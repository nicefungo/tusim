# FP32 softmax SRAM requests

The standalone descriptor selects how each buffered row is transferred, not how its softmax is computed. Zero-initialize the descriptor: `TU_SOFTMAX_SRAM_NATIVE` is the default and requires four-byte bank words. `TU_SOFTMAX_SRAM_ELEMENT` issues an explicitly sized FP32 request; `TU_SOFTMAX_SRAM_ROW` issues a contiguous request per row per direction. Neither combines adjacent rows.

In the [admitted-case matrix](evidence/softmax-sram-access/matrix.csv), two rows of three FP32 values move 24 bytes in and 24 bytes out. With eight-byte words, address-span metering and a one-byte displacement, element requests charge nine words per direction and 32 local penalty cycles; row requests charge five and 16. That is a difference in SRAM accounting, not a reduction in payload, a change in precision, or a measured speedup.

`make test-softmax-sram-access` checks the word and penalty counters, outputs and row statistics against independent calculations on initialized regions. It exercises standard, log and online modes, disjoint buffers, overlap in both directions, and in-place output. The contract is for this engine, not all SRAM clients.

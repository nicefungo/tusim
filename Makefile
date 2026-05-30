# TinyTU ONNX Compiler — Makefile
# ================================

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

TU_DIR     = tu_cmodel
COMPILER   = compiler/onnx_to_tu.py

.PHONY: all clean test test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-compiler test-asm test-memhier test-norm test-elementwise test-bf16 test-int-quant test-conv

all: libtucmodel.a

# ---- TU CModel library ----
TU_OBJS = $(TU_DIR)/tu_cmodel.o $(TU_DIR)/tu_asm.o $(TU_DIR)/tu_precision.o \
          $(TU_DIR)/tu_sram.o $(TU_DIR)/tu_dma.o $(TU_DIR)/dma_descriptor.o \
          $(TU_DIR)/rounding.o $(TU_DIR)/fp8.o \
          $(TU_DIR)/command_queue.o $(TU_DIR)/memory/dram_model.o \
          $(TU_DIR)/memory/memory_hierarchy.o \
          $(TU_DIR)/isa/tu_isa.o $(TU_DIR)/compute/elementwise_pipeline.o \
          $(TU_DIR)/compute/normalization_engine.o \
          $(TU_DIR)/compute/softmax_engine.o \
          $(TU_DIR)/compute/convolution_engine.o \
          $(TU_DIR)/infra/logging.o \
          $(TU_DIR)/tu_int_quant.o \
          $(TU_DIR)/compute/dataflow/dataflow_registry.o \
          $(TU_DIR)/compute/dataflow/dataflow_dispatcher.o \
          $(TU_DIR)/compute/dataflow/weight_stationary.o \
          $(TU_DIR)/compute/dataflow/output_stationary.o

libtucmodel.a: $(TU_OBJS)
	$(AR) rcs $@ $^

$(TU_DIR)/tu_cmodel.o: $(TU_DIR)/tu_cmodel.c $(TU_DIR)/tu_cmodel.h $(TU_DIR)/tu_precision.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_dma.h $(TU_DIR)/command_queue.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_asm.o: $(TU_DIR)/tu_asm.c $(TU_DIR)/tu_cmodel.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_precision.o: $(TU_DIR)/tu_precision.c $(TU_DIR)/tu_precision.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_sram.o: $(TU_DIR)/tu_sram.c $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_dma.o: $(TU_DIR)/tu_dma.c $(TU_DIR)/tu_dma.h $(TU_DIR)/dma_descriptor.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/dma_descriptor.o: $(TU_DIR)/dma_descriptor.c $(TU_DIR)/dma_descriptor.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/rounding.o: $(TU_DIR)/rounding.c $(TU_DIR)/rounding.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/fp8.o: $(TU_DIR)/fp8.c $(TU_DIR)/fp8.h $(TU_DIR)/rounding.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/command_queue.o: $(TU_DIR)/command_queue.c $(TU_DIR)/command_queue.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/compute/elementwise_pipeline.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/memory/dram_model.o: $(TU_DIR)/memory/dram_model.c $(TU_DIR)/memory/dram_model.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/memory/memory_hierarchy.o: $(TU_DIR)/memory/memory_hierarchy.c $(TU_DIR)/memory/memory_hierarchy.h $(TU_DIR)/memory/dram_model.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/isa/tu_isa.o: $(TU_DIR)/isa/tu_isa.c $(TU_DIR)/isa/tu_isa.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_precision.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/elementwise_pipeline.o: $(TU_DIR)/compute/elementwise_pipeline.c $(TU_DIR)/compute/elementwise_pipeline.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/normalization_engine.o: $(TU_DIR)/compute/normalization_engine.c $(TU_DIR)/compute/normalization_engine.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Compute: Softmax Engine (O7) ----
$(TU_DIR)/compute/softmax_engine.o: $(TU_DIR)/compute/softmax_engine.c $(TU_DIR)/compute/softmax_engine.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Compute: Convolution Engine (O2) ----
$(TU_DIR)/compute/convolution_engine.o: $(TU_DIR)/compute/convolution_engine.c $(TU_DIR)/compute/convolution_engine.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Precision: INT8 Quantization (D2) ----
$(TU_DIR)/tu_int_quant.o: $(TU_DIR)/tu_int_quant.c $(TU_DIR)/tu_int_quant.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Infrastructure: Logging (Q2) ----
$(TU_DIR)/infra/logging.o: $(TU_DIR)/infra/logging.c $(TU_DIR)/infra/logging.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Dataflow plugins (A4) ----
$(TU_DIR)/compute/dataflow/dataflow_registry.o: $(TU_DIR)/compute/dataflow/dataflow_registry.c $(TU_DIR)/compute/dataflow/dataflow_registry.h $(TU_DIR)/compute/dataflow/dataflow_interface.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/dataflow/dataflow_dispatcher.o: $(TU_DIR)/compute/dataflow/dataflow_dispatcher.c $(TU_DIR)/compute/dataflow/dataflow_interface.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/dataflow/weight_stationary.o: $(TU_DIR)/compute/dataflow/weight_stationary.c $(TU_DIR)/compute/dataflow/dataflow_interface.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/dataflow/output_stationary.o: $(TU_DIR)/compute/dataflow/output_stationary.c $(TU_DIR)/compute/dataflow/dataflow_interface.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Test: cmodel correctness ----
test-cmodel: tests/test_cmodel.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-cmodel

# ---- Test: command queue ----
test-cmdq: tests/test_command_queue.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-cmdq

# ---- Test: DMA descriptor engine ----
test-dma: tests/test_dma.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-dma

# ---- Test: DRAM model ----
test-dram: tests/test_dram.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-dram

# ---- Test: ISA definitions ----
test-isa: tests/test_isa.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-isa

# ---- Test: Golden reference verification ----
test-golden: tests/test_golden.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-golden --quick

# ---- Test: Elementwise pipeline ----
test-elementwise: tests/test_elementwise.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-elementwise

# ---- Test: BF16 + Subnormal handling ----
test-bf16: tests/test_bf16_subnormal.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-bf16

# ---- Test: Memory Hierarchy ----
test-memhier: tests/test_memory_hierarchy.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-memhier

# ---- Test: Normalization Engine ----
test-norm: tests/test_normalization.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-norm

# ---- Test: Dataflow plugins (A4) ----
test-dataflow: tests/test_dataflow.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-dataflow

# ---- Test: Structured Logging (Q2) ----
test-logging: tests/test_logging.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-logging

# ---- Test: INT8 Quantization (D2) ----
test-int-quant: tests/test_int_quant.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-int-quant

# ---- Test: Convolution Engine (O2) ----
test-conv: tests/test_convolution.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-conv

# ---- Test: Rounding Modes (D6) ----
test-rounding: tests/test_rounding.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-rounding

# ---- Test: FP8 E4M3/E5M2 (D4) ----
test-fp8: tests/test_fp8.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-fp8

test-golden-full: tests/test_golden.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-golden

# ---- Test: compile ONNX → TU ----
test-compiler:
	@echo "=== Compiling GPT-block ONNX → TU ==="
	python3 $(COMPILER) examples/gpt_block.onnx -o /tmp/gpt_block_tu.c -n gpt_block 2>&1
	@echo ""
	@echo "=== Lines of generated C ==="
	@wc -l /tmp/gpt_block_tu.c

# ---- Test: ASM interpreter ----
test-asm: libtucmodel.a
	@echo "=== Building ASM test ==="
	$(CC) $(CFLAGS) -I. -o /tmp/test_asm tests/test_asm.c -L. -ltucmodel $(LDFLAGS)
	@echo "=== Running ASM interpreter ==="
	/tmp/test_asm

# ══════════════════════════════════════════════════════════════════
# Test Suite — Comprehensive (Gap V3: CI/Regression Framework)
# ══════════════════════════════════════════════════════════════════

# Run full test suite: build library + all unit tests + integration
.PHONY: test test-quick test-random test-full
test: test-cmodel test-cmdq test-dma test-dram test-isa test-golden \
      test-elementwise test-bf16 test-memhier test-norm test-dataflow \
      test-logging test-int-quant test-conv test-asm test-rounding test-fp8
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo "  ✅ All tests complete"
	@echo "═══════════════════════════════════════════"

# Quick smoke test (pre-commit)
test-quick: test-cmodel test-cmdq test-dma test-asm
	@echo ""
	@echo "═══ Quick smoke test passed ═══"

# Extended random differential testing (nightly)
test-random: tests/test_random.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-random

# Full pipeline: ONNX → compile → run
test-full: test-compiler libtucmodel.a
	@echo ""
	@echo "=== Building generated code ==="
	$(CC) $(CFLAGS) -o /tmp/gpt_block_tu /tmp/gpt_block_tu.c -I. -L. -ltucmodel $(LDFLAGS) 2>&1 || true
	@echo ""
	@echo "=== Running on TU CModel ==="
	/tmp/gpt_block_tu 2>&1 || true

# ---- Clean ----
clean:
	rm -f $(TU_DIR)/*.o $(TU_DIR)/memory/*.o $(TU_DIR)/isa/*.o $(TU_DIR)/compute/*.o $(TU_DIR)/compute/dataflow/*.o $(TU_DIR)/infra/*.o libtucmodel.a
	rm -f test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-golden-full
	rm -f test-dataflow test-elementwise test-bf16 test-memhier test-norm test-logging
	rm -f test-int-quant test-conv test-random
	rm -f test-rounding
	rm -f test-fp8
	rm -f /tmp/gpt_block_tu /tmp/gpt_block_tu.c /tmp/test_asm
	rm -rf build/ci_reports

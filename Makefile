# TinyTU ONNX Compiler — Makefile
# ================================

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

TU_DIR     = tu_cmodel
COMPILER   = compiler/onnx_to_tu.py

.PHONY: all clean test test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-compiler test-asm

all: libtucmodel.a

# ---- TU CModel library ----
TU_OBJS = $(TU_DIR)/tu_cmodel.o $(TU_DIR)/tu_asm.o $(TU_DIR)/tu_precision.o \
          $(TU_DIR)/tu_sram.o $(TU_DIR)/tu_dma.o $(TU_DIR)/dma_descriptor.o \
          $(TU_DIR)/command_queue.o $(TU_DIR)/memory/dram_model.o \
          $(TU_DIR)/isa/tu_isa.o $(TU_DIR)/compute/elementwise_pipeline.o

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

$(TU_DIR)/command_queue.o: $(TU_DIR)/command_queue.c $(TU_DIR)/command_queue.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/compute/elementwise_pipeline.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/memory/dram_model.o: $(TU_DIR)/memory/dram_model.c $(TU_DIR)/memory/dram_model.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/isa/tu_isa.o: $(TU_DIR)/isa/tu_isa.c $(TU_DIR)/isa/tu_isa.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_precision.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/compute/elementwise_pipeline.o: $(TU_DIR)/compute/elementwise_pipeline.c $(TU_DIR)/compute/elementwise_pipeline.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
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

# ---- Full pipeline test: CModel → compile → run ----
test-full: test-compiler libtucmodel.a
	@echo ""
	@echo "=== Building generated code ==="
	$(CC) $(CFLAGS) -o /tmp/gpt_block_tu /tmp/gpt_block_tu.c -I. -L. -ltucmodel $(LDFLAGS) 2>&1 || true
	@echo ""
	@echo "=== Running on TU CModel ==="
	/tmp/gpt_block_tu 2>&1 || true

# ---- Test: ASM interpreter ----
test-asm: libtucmodel.a
	@echo "=== Building ASM test ==="
	$(CC) $(CFLAGS) -I. -o /tmp/test_asm tests/test_asm.c -L. -ltucmodel $(LDFLAGS)
	@echo "=== Running ASM interpreter ==="
	/tmp/test_asm

# ---- Clean ----
clean:
	rm -f $(TU_DIR)/*.o $(TU_DIR)/memory/*.o $(TU_DIR)/isa/*.o $(TU_DIR)/compute/*.o libtucmodel.a
	rm -f test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-golden-full /tmp/gpt_block_tu /tmp/gpt_block_tu.c

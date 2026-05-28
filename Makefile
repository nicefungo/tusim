# TinyTU ONNX Compiler — Makefile
# ================================

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

TU_DIR     = tu_cmodel
COMPILER   = compiler/onnx_to_tu.py

.PHONY: all clean test test-cmodel test-compiler

all: libtucmodel.a

# ---- TU CModel library ----
libtucmodel.a: $(TU_DIR)/tu_cmodel.o $(TU_DIR)/tu_asm.o $(TU_DIR)/tu_precision.o $(TU_DIR)/tu_sram.o $(TU_DIR)/tu_dma.o
	$(AR) rcs $@ $^

$(TU_DIR)/tu_cmodel.o: $(TU_DIR)/tu_cmodel.c $(TU_DIR)/tu_cmodel.h $(TU_DIR)/tu_precision.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_asm.o: $(TU_DIR)/tu_asm.c $(TU_DIR)/tu_cmodel.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_precision.o: $(TU_DIR)/tu_precision.c $(TU_DIR)/tu_precision.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_sram.o: $(TU_DIR)/tu_sram.c $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(TU_DIR)/tu_dma.o: $(TU_DIR)/tu_dma.c $(TU_DIR)/tu_dma.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ---- Test: cmodel correctness ----
test-cmodel: tests/test_cmodel.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-cmodel

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
	rm -f $(TU_DIR)/*.o libtucmodel.a
	rm -f test-cmodel /tmp/gpt_block_tu /tmp/gpt_block_tu.c

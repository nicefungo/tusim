# TinyTU ONNX Compiler — Makefile
# ================================

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -fPIC
LDFLAGS ?= -lm

TU_DIR     = tu_cmodel
COMPILER   = compiler/onnx_to_tu.py

.PHONY: all clean test test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-compiler test-asm test-memhier test-norm test-elementwise test-bf16 test-int-quant test-conv test-attention test-perf test-pool test-pipeline test-agen test-multicore test-multicast test-scatter-gather test-trace test-errors test-config test-dataflow test-logging test-rounding test-fp8 test-softmax test-double test-random test-full test-context test-compress test-scheduler test-liveness test-tf32 test-bench test-power test-debug test-dataflow-sweep test-rounding-sweep test-attention-sweep test-pooling-sweep test-softmax-sweep test-conv-sweep

all: libtucmodel.a libtucmodel.so

# ---- TU CModel library ----
TU_OBJS = $(TU_DIR)/tu_cmodel.o $(TU_DIR)/tu_asm.o $(TU_DIR)/tu_precision.o \
          $(TU_DIR)/tu_sram.o $(TU_DIR)/tu_dma.o $(TU_DIR)/dma_descriptor.o \
          $(TU_DIR)/rounding.o $(TU_DIR)/fp8.o $(TU_DIR)/tf32.o \
          $(TU_DIR)/command_queue.o $(TU_DIR)/memory/dram_model.o \
          $(TU_DIR)/memory/memory_hierarchy.o $(TU_DIR)/memory/double_buffer.o \
          $(TU_DIR)/memory/address_generator.o \
          $(TU_DIR)/isa/tu_isa.o $(TU_DIR)/compute/elementwise_pipeline.o \
          $(TU_DIR)/compute/normalization_engine.o \
          $(TU_DIR)/compute/softmax_engine.o \
          $(TU_DIR)/compute/convolution_engine.o \
          $(TU_DIR)/compute/attention_engine.o \
          $(TU_DIR)/compute/pooling_engine.o \
          $(TU_DIR)/infra/logging.o \
          $(TU_DIR)/infra/json_reader.o \
          $(TU_DIR)/infra/config.o \
          $(TU_DIR)/tu_int_quant.o \
          $(TU_DIR)/perf/performance_counters.o \
          $(TU_DIR)/perf/event_trace.o \
          $(TU_DIR)/perf/power_model.o \
          $(TU_DIR)/compute/pipeline_controller.o \
          $(TU_DIR)/compute/dataflow/dataflow_registry.o \
          $(TU_DIR)/compute/dataflow/dataflow_dispatcher.o \
          $(TU_DIR)/compute/dataflow/weight_stationary.o \
          $(TU_DIR)/compute/dataflow/output_stationary.o \
          $(TU_DIR)/compute/dataflow/row_stationary.o \
          $(TU_DIR)/tu_core.o $(TU_DIR)/tu_cluster.o \
          $(TU_DIR)/tu_status.o $(TU_DIR)/infra/tu_context.o \
          $(TU_DIR)/memory/weight_compress.o \
          $(TU_DIR)/isa/tu_scheduler.o \
          $(TU_DIR)/isa/tu_liveness.o \
          $(TU_DIR)/infra/tu_debug.o \
          $(TU_DIR)/bindings/tu_dpi.o

libtucmodel.a: $(TU_OBJS)
	$(AR) rcs $@ $^

libtucmodel.so: $(TU_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

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

$(TU_DIR)/tf32.o: $(TU_DIR)/tf32.c $(TU_DIR)/tf32.h $(TU_DIR)/rounding.h $(TU_DIR)/tu_precision.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/command_queue.o: $(TU_DIR)/command_queue.c $(TU_DIR)/command_queue.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/compute/elementwise_pipeline.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/memory/dram_model.o: $(TU_DIR)/memory/dram_model.c $(TU_DIR)/memory/dram_model.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/memory/memory_hierarchy.o: $(TU_DIR)/memory/memory_hierarchy.c $(TU_DIR)/memory/memory_hierarchy.h $(TU_DIR)/memory/dram_model.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Memory: Double Buffering (A7) ----
$(TU_DIR)/memory/double_buffer.o: $(TU_DIR)/memory/double_buffer.c $(TU_DIR)/memory/double_buffer.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Memory: Address Generator (M3) ----
$(TU_DIR)/memory/address_generator.o: $(TU_DIR)/memory/address_generator.c $(TU_DIR)/memory/address_generator.h $(TU_DIR)/tu_config.h $(TU_DIR)/dma_descriptor.h
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

# ---- Compute: Attention Engine (O3) ----
$(TU_DIR)/compute/attention_engine.o: $(TU_DIR)/compute/attention_engine.c $(TU_DIR)/compute/attention_engine.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/compute/dataflow/dataflow_interface.h $(TU_DIR)/compute/elementwise_pipeline.h $(TU_DIR)/compute/softmax_engine.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Compute: Pooling Engine (O6) ----
$(TU_DIR)/compute/pooling_engine.o: $(TU_DIR)/compute/pooling_engine.c $(TU_DIR)/compute/pooling_engine.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_precision.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Precision: INT8 Quantization (D2) ----
$(TU_DIR)/tu_int_quant.o: $(TU_DIR)/tu_int_quant.c $(TU_DIR)/tu_int_quant.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Infrastructure: Logging (Q2) ----
$(TU_DIR)/infra/logging.o: $(TU_DIR)/infra/logging.c $(TU_DIR)/infra/logging.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Infrastructure: JSON Reader (A1) ----
$(TU_DIR)/infra/json_reader.o: $(TU_DIR)/infra/json_reader.c $(TU_DIR)/infra/json_reader.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Infrastructure: Config Loader (A1) ----
$(TU_DIR)/infra/config.o: $(TU_DIR)/infra/config.c $(TU_DIR)/infra/config.h $(TU_DIR)/infra/json_reader.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Perf: Performance Counters (E4, P2.5 foundation) ----

# ---- Perf: Performance Counters (E4, P2.5 foundation) ----
$(TU_DIR)/perf/performance_counters.o: $(TU_DIR)/perf/performance_counters.c $(TU_DIR)/perf/performance_counters.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Perf: Event Tracing (P2.7) — VCD waveform generation for GTKWave ----
$(TU_DIR)/perf/event_trace.o: $(TU_DIR)/perf/event_trace.c $(TU_DIR)/perf/event_trace.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Perf: Power/Energy Model (E4) — configurable technology nodes, CACTI-derived energy tables
$(TU_DIR)/perf/power_model.o: $(TU_DIR)/perf/power_model.c $(TU_DIR)/perf/power_model.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Pipeline Controller (E2) — Software pipelining for DMA/compute overlap ----
$(TU_DIR)/compute/pipeline_controller.o: $(TU_DIR)/compute/pipeline_controller.c $(TU_DIR)/compute/pipeline_controller.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h $(TU_DIR)/dma_descriptor.h $(TU_DIR)/command_queue.h $(TU_DIR)/memory/double_buffer.h
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

$(TU_DIR)/compute/dataflow/row_stationary.o: $(TU_DIR)/compute/dataflow/row_stationary.c $(TU_DIR)/compute/dataflow/dataflow_interface.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Multi-core: tu_core and tu_cluster (A5) ----
$(TU_DIR)/tu_core.o: $(TU_DIR)/tu_core.c $(TU_DIR)/tu_core.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

$(TU_DIR)/tu_cluster.o: $(TU_DIR)/tu_cluster.c $(TU_DIR)/tu_cluster.h $(TU_DIR)/tu_core.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_sram.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Exception handling (E5) ----
$(TU_DIR)/tu_status.o: $(TU_DIR)/tu_status.c $(TU_DIR)/tu_status.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Multi-context execution (E3) ----
$(TU_DIR)/infra/tu_context.o: $(TU_DIR)/infra/tu_context.c $(TU_DIR)/infra/tu_context.h $(TU_DIR)/tu_core.h $(TU_DIR)/infra/config.h $(TU_DIR)/tu_status.h $(TU_DIR)/tu_precision.h $(TU_DIR)/rounding.h $(TU_DIR)/perf/performance_counters.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Memory: Weight Compression (M5) ----
$(TU_DIR)/memory/weight_compress.o: $(TU_DIR)/memory/weight_compress.c $(TU_DIR)/memory/weight_compress.h $(TU_DIR)/tu_config.h $(TU_DIR)/tu_precision.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Compiler: Scheduling Pass (C2) ----
$(TU_DIR)/isa/tu_scheduler.o: $(TU_DIR)/isa/tu_scheduler.c $(TU_DIR)/isa/tu_scheduler.h $(TU_DIR)/isa/tu_isa.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Compiler: Liveness Allocator (C3) ----
$(TU_DIR)/isa/tu_liveness.o: $(TU_DIR)/isa/tu_liveness.c $(TU_DIR)/isa/tu_liveness.h $(TU_DIR)/isa/tu_scheduler.h $(TU_DIR)/isa/tu_isa.h $(TU_DIR)/tu_config.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- Debug/Observability (I3) ----
$(TU_DIR)/infra/tu_debug.o: $(TU_DIR)/infra/tu_debug.c $(TU_DIR)/infra/tu_debug.h $(TU_DIR)/tu_core.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/tu_sram.h $(TU_DIR)/tu_dma.h $(TU_DIR)/command_queue.h $(TU_DIR)/perf/performance_counters.h $(TU_DIR)/infra/config.h $(TU_DIR)/infra/logging.h $(TU_DIR)/tu_status.h
	$(CC) $(CFLAGS) -I$(TU_DIR) -c -o $@ $<

# ---- DPI-C Bindings (I1) ----
$(TU_DIR)/bindings/tu_dpi.o: $(TU_DIR)/bindings/tu_dpi.c $(TU_DIR)/bindings/tu_dpi.h $(TU_DIR)/tu_cmodel.h $(TU_DIR)/tu_precision.h $(TU_DIR)/tu_sram.h $(TU_DIR)/compute/elementwise_pipeline.h $(TU_DIR)/compute/softmax_engine.h $(TU_DIR)/compute/normalization_engine.h $(TU_DIR)/infra/logging.h
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

# ---- Test: TF32 (D3) ----
test-tf32: tests/test_tf32.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-tf32

# ---- Test: Attention Engine (O3) ----
test-attention: tests/test_attention.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-attention

# ---- Test: Performance Counters (E4, P2.5) ----
test-perf: tests/test_perf_counters.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -Itests -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-perf

# ---- Test: Pooling Engine (O6) ----
test-pool: tests/test_pooling.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -Itests -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-pool

# ---- Test: Software Pipelining Controller (E2) ----
test-pipeline: tests/test_pipeline.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-pipeline

# ---- Test: Address Generator (M3) ----
test-agen: tests/test_address_gen.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-agen

# ---- Test: Power/Energy Model (E4) ----
test-power: tests/test_power_model.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-power

# ---- Test: Debug & Observability (I3) ----
test-debug: tests/test_debug.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-debug

# ---- Test: DPI-C Integration (I1) ----
test-dpi: tests/test_dpi.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-dpi

# ---- Test: Multi-core Cluster (A5) ----
test-multicore: tests/test_multicore.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-multicore

# ---- Test: Multicast/Broadcast DMA (DM4) ----
test-multicast: tests/test_multicast.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-multicast

# ---- Test: Scatter/Gather DMA (DM3) ----
test-scatter-gather: tests/test_scatter_gather.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-scatter-gather

# ---- Benchmark: Comparative Performance (P2.9/V5) ----
test-bench: tests/test_benchmark.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-bench

# ---- Test: Dataflow Sweep (exploration harness) ----
test-dataflow-sweep: tests/test_dataflow_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-dataflow-sweep

# ---- Test: Rounding Mode Sweep (exploration harness) ----
test-rounding-sweep: tests/test_rounding_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-rounding-sweep

# ---- Test: Attention Engine Sweep (exploration harness) ----
test-attention-sweep: tests/test_attention_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-attention-sweep

# ---- Test: Pooling Engine Sweep (exploration harness) ----
test-pooling-sweep: tests/test_pooling_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-pooling-sweep

# ---- Test: Softmax Mode Sweep (exploration harness) ----
test-softmax-sweep: tests/test_softmax_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-softmax-sweep

# ---- Test: Convolution Kernel×Stride Sweep (exploration harness) ----
test-conv-sweep: tests/test_conv_sweep.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-conv-sweep

# ---- Test: Weight Compression (M5) ----
test-compress: tests/test_compress.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -Itests -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-compress

# ---- Test: Compiler Scheduling Pass (C2) ----
test-scheduler: tests/test_scheduler.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-scheduler

# ---- Test: Liveness Allocator (C3) ----
test-liveness: tests/test_liveness.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-liveness

# ---- Test: Multi-context Execution (E3) ----
test-context: tests/test_context.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-context test-compress

# ---- Test: Event Tracing (P2.7) — VCD waveform generation ----
test-trace: tests/test_trace.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-trace

# ---- Test: Error Handling (E5) ----
test-errors: tests/test_error_handling.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-errors

# ---- Test: JSON Config Loader (A1) ----
test-config: tests/test_config.c libtucmodel.a
	$(CC) $(CFLAGS) -I. -Itu_cmodel -o $@ $< -L. -ltucmodel $(LDFLAGS)
	./test-config

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
      test-logging test-int-quant test-conv test-asm test-rounding test-fp8 \
      test-attention test-perf test-pool test-pipeline test-agen test-multicore \
      test-multicast test-scatter-gather test-trace test-config test-scheduler test-liveness test-dpi
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

# ---- Documentation ----#
# Generate Doxygen API documentation
docs-api:
	@command -v doxygen >/dev/null 2>&1 || { echo "doxygen not found. Install: sudo apt install doxygen graphviz"; exit 1; }
	@mkdir -p docs/api
	doxygen Doxyfile 2>&1 | grep -v "^$$" || true
	@echo "Docs generated in docs/api/html/index.html"

# Generate auto-config reference markdown
config-docs: libtucmodel.a
	@echo "Generating config reference documentation..."
	@printf '#include "tu_cmodel/infra/config.h"\nint main(void) { tu_config_t cfg; tu_config_default(&cfg); tu_config_emit_docs(&cfg, stdout); return 0; }\n' > /tmp/tu_config_docs.c
	@$(CC) $(CFLAGS) -I. -Itu_cmodel -o /tmp/tu_config_docs /tmp/tu_config_docs.c -L. -ltucmodel $(LDFLAGS)
	/tmp/tu_config_docs > docs/CONFIG_REFERENCE.md
	@rm -f /tmp/tu_config_docs /tmp/tu_config_docs.c
	@echo "Config reference written to docs/CONFIG_REFERENCE.md"

# ---- Clean ----
clean:
	rm -f $(TU_DIR)/*.o $(TU_DIR)/memory/*.o $(TU_DIR)/isa/*.o $(TU_DIR)/compute/*.o $(TU_DIR)/compute/dataflow/*.o $(TU_DIR)/infra/*.o $(TU_DIR)/perf/*.o $(TU_DIR)/bindings/*.o libtucmodel.a libtucmodel.so
	rm -f test-cmodel test-cmdq test-dma test-dram test-isa test-golden test-golden-full
	rm -f test-dataflow test-elementwise test-bf16 test-memhier test-norm test-logging
	rm -f test-int-quant test-conv test-random
	rm -f test-rounding test-fp8 test-attention test-perf test-pool test-pipeline
	rm -f test-agen test-multicore test-compress test-errors test-config test-softmax test-double test-context test-compress test-scheduler test-liveness test-power test-dpi test-dataflow-sweep test-rounding-sweep test-attention-sweep test-pooling-sweep test-softmax-sweep test-conv-sweep
	rm -f /tmp/gpt_block_tu /tmp/gpt_block_tu.c /tmp/test_asm
	rm -rf build/ci_reports

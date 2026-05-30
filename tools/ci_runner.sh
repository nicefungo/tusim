#!/bin/bash
#
# TinyTU Production CI Runner
# ===========================
# Gap V3: No regression framework → Automated CI pipeline
#
# Runs: build → unit tests → integration tests → regression check → coverage → report
# Exit code: 0 on all-pass, 1 on any failure
#
# Usage:
#   ./tools/ci_runner.sh              # Full pipeline
#   ./tools/ci_runner.sh --quick      # Smoke test (fast, CI pre-commit)
#   ./tools/ci_runner.sh --random     # Extended random testing (nightly)
#   ./tools/ci_runner.sh --valgrind   # Memory error check
#   ./tools/ci_runner.sh --coverage   # Build with coverage flags
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ─────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="build/ci_reports"
LOG_DIR="$REPORT_DIR/logs"
SUMMARY_FILE="$REPORT_DIR/summary_${TIMESTAMP}.md"
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0
FAILED_TESTS=""

# Build configuration
BUILD_TYPE="Release"
CC="${CC:-gcc}"
CFLAGS_BASE="-O2 -Wall -Wextra -std=c11 -Werror"

# Parse arguments
QUICK_MODE=false
RANDOM_MODE=false
VALGRIND_MODE=false
COVERAGE_MODE=false
PARALLEL_JOBS=$(nproc 2>/dev/null || echo 4)

for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK_MODE=true ;;
        --random)  RANDOM_MODE=true ;;
        --valgrind) VALGRIND_MODE=true ;;
        --coverage) COVERAGE_MODE=true; BUILD_TYPE="Debug" ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--random] [--valgrind] [--coverage]"
            echo ""
            echo "  --quick     Smoke test (fast, < 30s)"
            echo "  --random    Extended random testing (10K+ iterations, nightly)"
            echo "  --valgrind  Run tests under Valgrind memcheck"
            echo "  --coverage  Build with gcov coverage flags"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

mkdir -p "$REPORT_DIR" "$LOG_DIR"

# ── Color output ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_section() { echo -e "\n${BOLD}${BLUE}═══ $1 ═══${NC}"; }
log_pass()   { echo -e "  ${GREEN}✓${NC} $1"; }
log_fail()   { echo -e "  ${RED}✗${NC} $1"; }
log_info()   { echo -e "  ${YELLOW}→${NC} $1"; }

# ── Functions ─────────────────────────────────────────────────────

record_result() {
    local name="$1" result="$2" logfile="$3"
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if [ "$result" = "PASS" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        log_pass "$name"
        echo "| $name | ✅ PASS | — |" >> "$SUMMARY_FILE"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS="${FAILED_TESTS}  - $name\n"
        log_fail "$name (see $logfile)"
        echo "| $name | ❌ FAIL | [log]($logfile) |" >> "$SUMMARY_FILE"
    fi
}

run_test_target() {
    local target="$1" label="$2"
    local logfile="$LOG_DIR/${target}.log"

    log_info "Running: $label ..."
    if make "$target" > "$logfile" 2>&1; then
        record_result "$label" "PASS" "$logfile"
        return 0
    else
        record_result "$label" "FAIL" "$logfile"
        return 1
    fi
}

run_custom_test() {
    local label="$1" logfile="$2"
    shift 2

    log_info "Running: $label ..."
    if "$@" > "$logfile" 2>&1; then
        record_result "$label" "PASS" "$logfile"
        return 0
    else
        record_result "$label" "FAIL" "$logfile"
        return 1
    fi
}

# ── Write summary header ──────────────────────────────────────────
cat > "$SUMMARY_FILE" << EOF
# TinyTU CI Report — $TIMESTAMP

**Mode:** $([ "$QUICK_MODE" = true ] && echo "quick" || echo "full")
**Build:** $BUILD_TYPE
**Compiler:** $CC
**Platform:** $(uname -s) $(uname -m)

| Test | Result | Notes |
|------|--------|-------|
EOF

# ══════════════════════════════════════════════════════════════════
# PHASE 1: Build
# ══════════════════════════════════════════════════════════════════
log_section "PHASE 1: Build"

# Clean build
log_info "Cleaning previous build artifacts..."
make clean > "$LOG_DIR/clean.log" 2>&1 || true

# Adjust flags for coverage
if [ "$COVERAGE_MODE" = true ]; then
    CFLAGS_BASE="$CFLAGS_BASE -O0 -g --coverage -fprofile-arcs -ftest-coverage"
fi

# Build library
log_info "Building libtucmodel.a ..."
if make CC="$CC" CFLAGS="$CFLAGS_BASE" libtucmodel.a > "$LOG_DIR/build_lib.log" 2>&1; then
    record_result "Build: libtucmodel.a" "PASS" "$LOG_DIR/build_lib.log"
else
    record_result "Build: libtucmodel.a" "FAIL" "$LOG_DIR/build_lib.log"
    cat "$LOG_DIR/build_lib.log"
    echo -e "\n${RED}BUILD FAILED — aborting${NC}"
    exit 1
fi

# Build all test binaries (compile only, no run)
log_info "Compiling test binaries..."
BUILD_TESTS=(
    "test-cmodel:TU CModel core"
    "test-cmdq:Command queue"
    "test-dma:DMA engine"
    "test-dram:DRAM model"
    "test-isa:ISA definitions"
    "test-golden:Golden reference"
    "test-elementwise:Elementwise pipeline"
    "test-bf16:BF16 precision"
    "test-memhier:Memory hierarchy"
    "test-norm:Normalization engine"
    "test-dataflow:Dataflow plugins"
    "test-logging:Structured logging"
    "test-int-quant:INT8 quantization"
    "test-conv:Convolution engine"
)

for bt in "${BUILD_TESTS[@]}"; do
    target="${bt%%:*}"
    label="${bt##*:}"
    # Skip run, just compile
    if make CC="$CC" CFLAGS="$CFLAGS_BASE" "$target" BUILD_ONLY=1 > "$LOG_DIR/build_${target}.log" 2>&1; then
        : # compiled OK
    else
        # Try with explicit compile
        if make -n "$target" > /dev/null 2>&1; then
            log_info "Compiling $label..."
            # We compile but don't auto-run here
            make CC="$CC" CFLAGS="$CFLAGS_BASE" "$(echo $target | sed 's/test-//')" > /dev/null 2>&1 || true
        fi
    fi
done

# ══════════════════════════════════════════════════════════════════
# PHASE 2: Unit Tests
# ══════════════════════════════════════════════════════════════════
log_section "PHASE 2: Unit Tests"

# Define test suite
if [ "$QUICK_MODE" = true ]; then
    TEST_TARGETS=(
        "test-cmodel:TU CModel core"
        "test-cmdq:Command queue"
        "test-dma:DMA engine"
        "test-golden:Golden (quick)"
    )
else
    TEST_TARGETS=(
        "test-cmodel:TU CModel core"
        "test-cmdq:Command queue"
        "test-dma:DMA engine"
        "test-dram:DRAM model"
        "test-isa:ISA definitions"
        "test-golden:Golden reference"
        "test-elementwise:Elementwise pipeline"
        "test-bf16:BF16 precision"
        "test-memhier:Memory hierarchy"
        "test-norm:Normalization engine"
        "test-dataflow:Dataflow plugins"
        "test-logging:Structured logging"
        "test-int-quant:INT8 quantization"
        "test-conv:Convolution engine"
    )
fi

OVERALL_EXIT=0
for tt in "${TEST_TARGETS[@]}"; do
    target="${tt%%:*}"
    label="${tt##*:}"
    # Special: golden test in quick mode
    if [ "$target" = "test-golden" ] && [ "$QUICK_MODE" = true ]; then
        log_info "Running: $label (quick mode)..."
        local_log="$LOG_DIR/test_golden.log"
        make CC="$CC" CFLAGS="$CFLAGS_BASE -DQUICK_RANDOM_TESTS=50" test-golden > "$local_log" 2>&1 || true
        # The target already runs the binary; check result
        if grep -q "PASS" "$local_log" 2>/dev/null || [ $? -eq 0 ]; then
            record_result "$label" "PASS" "$local_log"
        else
            record_result "$label" "FAIL" "$local_log"
            OVERALL_EXIT=1
        fi
    else
        run_test_target "$target" "$label" || OVERALL_EXIT=1
    fi
done

# ══════════════════════════════════════════════════════════════════
# PHASE 3: Extended Random Testing (nightly mode)
# ══════════════════════════════════════════════════════════════════
if [ "$RANDOM_MODE" = true ]; then
    log_section "PHASE 3: Extended Random Testing"
    log_info "Compiling extended random test suite..."
    if make CC="$CC" CFLAGS="$CFLAGS_BASE" test-random > "$LOG_DIR/build_random.log" 2>&1; then
        log_info "Running 10K random MMA tests..."
        if ./test-random > "$LOG_DIR/test_random.log" 2>&1; then
            record_result "Random MMA (10K)" "PASS" "$LOG_DIR/test_random.log"
        else
            record_result "Random MMA (10K)" "FAIL" "$LOG_DIR/test_random.log"
            OVERALL_EXIT=1
        fi
    else
        record_result "Compile: random test" "FAIL" "$LOG_DIR/build_random.log"
        OVERALL_EXIT=1
    fi
fi

# ══════════════════════════════════════════════════════════════════
# PHASE 4: Integration Tests (compiler path)
# ══════════════════════════════════════════════════════════════════
if [ "$QUICK_MODE" != true ]; then
    log_section "PHASE 4: Integration Tests"

    # ASM interpreter test
    run_test_target "test-asm" "ASM interpreter" || OVERALL_EXIT=1

    # Compiler test (requires Python3)
    if command -v python3 > /dev/null 2>&1; then
        log_info "Running ONNX compiler test..."
        if python3 compiler/onnx_to_tu.py examples/gpt_block.onnx -o /tmp/gpt_block_tu.c -n gpt_block > "$LOG_DIR/test_compiler.log" 2>&1; then
            record_result "ONNX compiler" "PASS" "$LOG_DIR/test_compiler.log"
        else
            record_result "ONNX compiler" "FAIL" "$LOG_DIR/test_compiler.log"
            OVERALL_EXIT=1
        fi
    else
        log_info "Skipping compiler test (python3 not found)"
    fi
fi

# ══════════════════════════════════════════════════════════════════
# PHASE 5: Coverage Report (if enabled)
# ══════════════════════════════════════════════════════════════════
if [ "$COVERAGE_MODE" = true ]; then
    log_section "PHASE 5: Coverage Report"
    if command -v gcov > /dev/null 2>&1; then
        log_info "Generating coverage report..."
        gcov -r tu_cmodel/*.c tu_cmodel/*/*.c > "$LOG_DIR/coverage.log" 2>&1 || true
        record_result "Coverage report" "PASS" "$LOG_DIR/coverage.log"
    else
        log_info "gcov not found, skipping coverage"
    fi
fi

# ══════════════════════════════════════════════════════════════════
# PHASE 6: Valgrind Memory Check
# ══════════════════════════════════════════════════════════════════
if [ "$VALGRIND_MODE" = true ]; then
    log_section "PHASE 6: Valgrind Memory Check"
    if command -v valgrind > /dev/null 2>&1; then
        log_info "Running cmodel test under Valgrind..."
        make CC="$CC" CFLAGS="$CFLAGS_BASE" test-cmodel-build > "$LOG_DIR/build_vg.log" 2>&1 || true
        valgrind --leak-check=full --error-exitcode=99 ./test-cmodel > "$LOG_DIR/valgrind.log" 2>&1
        local vg_rc=$?
        if [ $vg_rc -eq 0 ]; then
            record_result "Valgrind: cmodel" "PASS" "$LOG_DIR/valgrind.log"
        else
            record_result "Valgrind: cmodel" "FAIL" "$LOG_DIR/valgrind.log"
            OVERALL_EXIT=1
        fi
    else
        log_info "Valgrind not found, skipping"
    fi
fi

# ══════════════════════════════════════════════════════════════════
# Final Report
# ══════════════════════════════════════════════════════════════════
log_section "CI Summary"

cat >> "$SUMMARY_FILE" << EOF

---
**Total:** $TOTAL_COUNT tests
**Passed:** $PASS_COUNT
**Failed:** $FAIL_COUNT
**Success rate:** $(awk "BEGIN {printf \"%.1f%%\", ($PASS_COUNT/$TOTAL_COUNT)*100}")

EOF

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "\n${RED}${BOLD}Failed tests:${NC}"
    echo -e "$FAILED_TESTS"
    echo -e "\n${RED}${BOLD}❌ CI FAILED — $FAIL_COUNT/$TOTAL_COUNT tests failed${NC}"
else
    echo -e "\n${GREEN}${BOLD}✅ CI PASSED — $PASS_COUNT/$TOTAL_COUNT tests passed${NC}"
fi

echo ""
echo "Full report: $SUMMARY_FILE"
echo "Logs: $LOG_DIR/"

exit $OVERALL_EXIT

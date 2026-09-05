# Build/run for the AXI4-Lite <-> APB bridge + GPT timer project.
# Requires: verilator (5.020+), gtkwave (optional, for viewing waves.vcd).
#
#   make sim      - run the full directed test suite + the random test
#   make test TEST=seq_one_shot_timeout   - run one named test
#   make waves    - rerun one test with VCD tracing enabled (sim/waves.vcd)
#   make clean

VERILATOR   ?= verilator
TOP_DIR     := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
RTL         := $(TOP_DIR)/rtl/gpt_timer.sv $(TOP_DIR)/rtl/axi4lite_apb_bridge.sv
# Order matters: axi_lite_if.sv/apb_if.sv now embed the driver/monitor/
# ref-model loops directly (see those files' headers - a Verilator bug
# means a class task taking a `virtual interface` parameter can corrupt
# the whole design's combinational convergence, even unused), so the
# classes they reference (tb_base_pkg, axi_txn, apb_txn, gpt_ref_model)
# must be compiled first.
TB_SOURCES  := \
  $(TOP_DIR)/tb/base/tb_base_pkg.sv \
  $(TOP_DIR)/tb/agents/axi_lite/axi_txn.sv \
  $(TOP_DIR)/tb/agents/apb/apb_txn.sv \
  $(TOP_DIR)/tb/env/gpt_ref_model.sv \
  $(TOP_DIR)/tb/interfaces/axi_lite_if.sv \
  $(TOP_DIR)/tb/interfaces/apb_if.sv \
  $(TOP_DIR)/tb/agents/axi_lite/axi_driver.sv \
  $(TOP_DIR)/tb/agents/axi_lite/axi_monitor.sv \
  $(TOP_DIR)/tb/agents/axi_lite/axi_agent.sv \
  $(TOP_DIR)/tb/agents/apb/apb_monitor.sv \
  $(TOP_DIR)/tb/env/scoreboard.sv \
  $(TOP_DIR)/tb/env/coverage.sv \
  $(TOP_DIR)/tb/env/env.sv \
  $(TOP_DIR)/tb/seq/gpt_sequences.sv \
  $(TOP_DIR)/tb/tb_top.sv

OBJ_DIR     := obj_dir
BIN         := $(OBJ_DIR)/tb_sim
OBJ_DIR_WAV := obj_dir_waves
BIN_WAV     := $(OBJ_DIR_WAV)/tb_sim

VFLAGS := --binary --timing -Wno-fatal --assert -Wno-DECLFILENAME -Wno-TIMESCALEMOD

DIRECTED_TESTS := \
  seq_reset_and_id \
  seq_ctrl_load_rw \
  seq_one_shot_timeout \
  seq_status_w1c \
  seq_periodic_reload \
  seq_load_while_busy \
  seq_bad_address

.PHONY: all sim build build-waves waves test clean sanity skeleton

all: sim

build: $(RTL) $(TB_SOURCES)
	$(VERILATOR) $(VFLAGS) $(RTL) $(TB_SOURCES) --top-module tb_top -o tb_sim -Mdir $(OBJ_DIR)

sim: build
	@fail=0; \
	for t in $(DIRECTED_TESTS) seq_random_mix; do \
		echo "=== running $$t ==="; \
		./$(BIN) +TEST=$$t || fail=1; \
		echo; \
	done; \
	exit $$fail

test: build
	./$(BIN) +TEST=$(TEST)

build-waves: $(RTL) $(TB_SOURCES)
	$(VERILATOR) $(VFLAGS) --trace $(RTL) $(TB_SOURCES) --top-module tb_top -o tb_sim -Mdir $(OBJ_DIR_WAV)

waves: build-waves
	./$(BIN_WAV) +TEST=$(if $(TEST),$(TEST),seq_one_shot_timeout) +DUMP
	@echo "Wrote waves.vcd - view with: gtkwave waves.vcd"

sanity:
	$(VERILATOR) $(VFLAGS) $(RTL) $(TOP_DIR)/sim/sanity_tb.sv --top-module sanity_tb -o sanity_sim -Mdir obj_dir_sanity
	./obj_dir_sanity/sanity_sim

skeleton:
	$(VERILATOR) $(VFLAGS) $(TOP_DIR)/tb/base/tb_base_pkg.sv $(TOP_DIR)/sim/skeleton_tb.sv --top-module skeleton_tb -o skeleton_sim -Mdir obj_dir_skeleton
	./obj_dir_skeleton/skeleton_sim

clean:
	rm -rf obj_dir obj_dir_waves obj_dir_sanity obj_dir_skeleton *.vcd

`timescale 1ns/1ps
// Top-level testbench: instantiates the DUT (bridge + GPT timer), the TB
// interfaces, builds the environment, and runs one sequence selected by a
// +TEST= plusarg - the closest equivalent this hand-written framework has
// to `run_test("test_name")` with +UVM_TESTNAME=.
module tb_top;
  logic clk = 0;
  logic rstn = 0;
  always #5 clk = ~clk;

  axi_lite_if #(.ADDR_WIDTH(32)) axi_vif (.aclk(clk), .aresetn(rstn));
  apb_if      #(.ADDR_WIDTH(5))  apb_vif (.pclk(clk), .presetn(rstn));

  axi4lite_apb_bridge #(.AXI_ADDR_WIDTH(32), .APB_ADDR_WIDTH(5)) dut_bridge (
    .aclk(clk), .aresetn(rstn),
    .s_axi_awaddr(axi_vif.awaddr), .s_axi_awvalid(axi_vif.awvalid), .s_axi_awready(axi_vif.awready),
    .s_axi_wdata(axi_vif.wdata), .s_axi_wstrb(axi_vif.wstrb), .s_axi_wvalid(axi_vif.wvalid), .s_axi_wready(axi_vif.wready),
    .s_axi_bresp(axi_vif.bresp), .s_axi_bvalid(axi_vif.bvalid), .s_axi_bready(axi_vif.bready),
    .s_axi_araddr(axi_vif.araddr), .s_axi_arvalid(axi_vif.arvalid), .s_axi_arready(axi_vif.arready),
    .s_axi_rdata(axi_vif.rdata), .s_axi_rresp(axi_vif.rresp), .s_axi_rvalid(axi_vif.rvalid), .s_axi_rready(axi_vif.rready),
    .m_apb_psel(apb_vif.psel), .m_apb_penable(apb_vif.penable), .m_apb_pwrite(apb_vif.pwrite),
    .m_apb_paddr(apb_vif.paddr), .m_apb_pwdata(apb_vif.pwdata), .m_apb_prdata(apb_vif.prdata),
    .m_apb_pready(apb_vif.pready), .m_apb_pslverr(apb_vif.pslverr)
  );

  gpt_timer #(.ADDR_WIDTH(5)) dut_gpt (
    .pclk(clk), .presetn(rstn),
    .psel(apb_vif.psel), .penable(apb_vif.penable), .pwrite(apb_vif.pwrite),
    .paddr(apb_vif.paddr), .pwdata(apb_vif.pwdata),
    .prdata(apb_vif.prdata), .pready(apb_vif.pready), .pslverr(apb_vif.pslverr)
  );

  gpt_env env;

  initial begin
    // global watchdog: a stuck DUT/sequence fails loudly instead of hanging forever
    #2_000_000;
    $display("WATCHDOG TIMEOUT: tb_top did not finish in time");
    $finish;
  end

  initial begin
    if ($test$plusargs("DUMP")) begin
      $dumpfile("waves.vcd");
      $dumpvars(0, tb_top);
    end
  end

  initial begin
    string test_name;
    gpt_base_sequence seq;

    env = new("env", axi_vif, apb_vif);

    rstn = 0;
    repeat (5) @(posedge clk);
    rstn = 1;
    @(posedge clk);

    env.run(apb_vif);

    if (!$value$plusargs("TEST=%s", test_name)) test_name = "seq_reset_and_id";
    $display("[%0t] tb_top: running test '%s'", $time, test_name);

    case (test_name)
      "seq_reset_and_id":     seq = new_reset_and_id();
      "seq_ctrl_load_rw":     seq = new_ctrl_load_rw();
      "seq_one_shot_timeout": seq = new_one_shot_timeout();
      "seq_status_w1c":       seq = new_status_w1c();
      "seq_periodic_reload":  seq = new_periodic_reload();
      "seq_load_while_busy":  seq = new_load_while_busy();
      "seq_bad_address":      seq = new_bad_address();
      "seq_random_mix":       seq = new_random_mix();
      "seq_all_directed":     seq = null; // handled specially below
      default: begin
        $display("Unknown TEST '%s'", test_name);
        $finish;
      end
    endcase

    if (test_name == "seq_all_directed") begin
      run_seq(new_reset_and_id());
      run_seq(new_ctrl_load_rw());
      run_seq(new_one_shot_timeout());
      run_seq(new_status_w1c());
      run_seq(new_periodic_reload());
      run_seq(new_load_while_busy());
      run_seq(new_bad_address());
    end else begin
      run_seq(seq);
    end

    // allow any in-flight APB completion / scoreboard callback to settle
    repeat (5) @(posedge clk);

    tb_base_pkg::test_report::summary();
    env.sb.report();
    env.cov.report();
    $display("FUNCTIONAL COVERAGE: %0.2f%%", env.cov.get_coverage());

    if (tb_base_pkg::test_report::num_fails == 0 && env.sb.num_errors == 0)
      $display("TEST '%s': PASSED", test_name);
    else
      $display("TEST '%s': FAILED", test_name);

    $finish;
  end

  task automatic run_seq(gpt_base_sequence s);
    s.start(env.axi_agt.sqr);
  endtask

  function automatic seq_reset_and_id new_reset_and_id();
    seq_reset_and_id s = new();
    return s;
  endfunction
  function automatic seq_ctrl_load_rw new_ctrl_load_rw();
    seq_ctrl_load_rw s = new();
    return s;
  endfunction
  function automatic seq_one_shot_timeout new_one_shot_timeout();
    seq_one_shot_timeout s = new();
    return s;
  endfunction
  function automatic seq_status_w1c new_status_w1c();
    seq_status_w1c s = new();
    return s;
  endfunction
  function automatic seq_periodic_reload new_periodic_reload();
    seq_periodic_reload s = new();
    return s;
  endfunction
  function automatic seq_load_while_busy new_load_while_busy();
    seq_load_while_busy s = new();
    return s;
  endfunction
  function automatic seq_bad_address new_bad_address();
    seq_bad_address s = new();
    return s;
  endfunction
  function automatic seq_random_mix new_random_mix();
    seq_random_mix s = new();
    return s;
  endfunction

endmodule

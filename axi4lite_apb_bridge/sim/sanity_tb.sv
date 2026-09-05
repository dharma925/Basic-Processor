`timescale 1ns/1ps
// Stage-1 sanity check: a plain, class-free, task-based testbench exercising
// the bridge + GPT timer before any UVM-style infrastructure is built.
module sanity_tb;
  logic clk = 0;
  logic rstn = 0;
  always #5 clk = ~clk;

  logic [31:0] awaddr, wdata, araddr, rdata;
  logic [3:0]  wstrb = 4'hF;
  logic        awvalid, awready, wvalid, wready, bvalid, bready;
  logic [1:0]  bresp, rresp;
  logic        arvalid, arready, rvalid, rready;

  logic        psel, penable, pwrite, pready, pslverr;
  logic [4:0]  paddr;
  logic [31:0] pwdata, prdata;

  axi4lite_apb_bridge #(.AXI_ADDR_WIDTH(32), .APB_ADDR_WIDTH(5)) dut_bridge (
    .aclk(clk), .aresetn(rstn),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .m_apb_psel(psel), .m_apb_penable(penable), .m_apb_pwrite(pwrite),
    .m_apb_paddr(paddr), .m_apb_pwdata(pwdata), .m_apb_prdata(prdata),
    .m_apb_pready(pready), .m_apb_pslverr(pslverr)
  );

  gpt_timer #(.ADDR_WIDTH(5)) dut_gpt (
    .pclk(clk), .presetn(rstn),
    .psel(psel), .penable(penable), .pwrite(pwrite), .paddr(paddr), .pwdata(pwdata),
    .prdata(prdata), .pready(pready), .pslverr(pslverr)
  );

  // Drive convention used throughout this project's testbenches: stimulus is
  // applied via blocking assignment a small delta *after* the clock edge
  // (@(posedge clk); #1; sig = val;), and sampled at #1 too (equivalently,
  // negedge). Without a real clocking block this is the standard
  // simulator-portable way to avoid racing the DUT's own same-edge
  // nonblocking updates - it also sidesteps a Verilator-specific quirk where
  // nonblocking assignments inside an initial-rooted process/task execute as
  // if blocking (see the INITIALDLY warning).
  task automatic axi_write(input logic [31:0] addr, input logic [31:0] data);
    @(posedge clk); #1;
    awaddr = addr; awvalid = 1; wdata = data; wvalid = 1; bready = 1;
    @(negedge clk); // mid-cycle: safe point to sample ready, strictly before
    while (!(awready && wready)) @(negedge clk); // the capturing posedge
    @(posedge clk); #1; // capturing edge, +1ns so this write doesn't race the DUT's own NBA update on the same edge
    awvalid = 0; wvalid = 0;
    @(negedge clk);
    while (!bvalid) @(negedge clk);
    $display("[%0t] WRITE addr=%0h data=%0h resp=%0d", $time, addr, data, bresp);
    @(posedge clk); #1;
    bready = 0;
  endtask

  task automatic axi_read(input logic [31:0] addr, output logic [31:0] data, output logic [1:0] resp);
    @(posedge clk); #1;
    araddr = addr; arvalid = 1; rready = 1;
    @(negedge clk);
    while (!arready) @(negedge clk);
    @(posedge clk); #1; // capturing edge
    arvalid = 0;
    @(negedge clk);
    while (!rvalid) @(negedge clk);
    data = rdata; resp = rresp;
    $display("[%0t] READ  addr=%0h data=%0h resp=%0d", $time, addr, data, resp);
    @(posedge clk); #1;
    rready = 0;
  endtask

  logic [31:0] rd;
  logic [1:0]  rd_resp;
  int errors = 0;

  initial begin
    #100000;
    $display("WATCHDOG TIMEOUT: testbench did not finish in time");
    $finish;
  end

  initial begin
    awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
    awaddr=0; wdata=0; araddr=0;
    rstn = 0;
    repeat (3) @(posedge clk);
    rstn = 1;
    @(posedge clk);

    // ID read
    axi_read('h10, rd, rd_resp);
    if (rd !== 32'hC0FFEE01 || rd_resp !== 2'b00) begin errors++; $display("FAIL: ID mismatch"); end

    // CTRL/LOAD write-readback
    axi_write('h08, 32'd5); // LOAD = 5
    axi_read('h08, rd, rd_resp);
    if (rd !== 32'd5) begin errors++; $display("FAIL: LOAD readback"); end

    // start one-shot count
    axi_write('h00, 32'b001); // EN=1, MODE=0
    // wait for count to reach 0 and TO to be set: 5 decrements + 1 detect cycle
    repeat (10) @(posedge clk);
    axi_read('h04, rd, rd_resp); // STATUS
    if (rd[1] !== 1'b1) begin errors++; $display("FAIL: TO not set after one-shot, STATUS=%0h", rd); end
    if (rd[0] !== 1'b0) begin errors++; $display("FAIL: BUSY should be clear after one-shot, STATUS=%0h", rd); end

    // W1C
    axi_write('h04, 32'b10); // clear TO
    axi_read('h04, rd, rd_resp);
    if (rd[1] !== 1'b0) begin errors++; $display("FAIL: TO not cleared by W1C"); end

    // writing 0 to STATUS has no effect
    axi_write('h00, 32'b001); // restart one-shot
    repeat (10) @(posedge clk);
    axi_write('h04, 32'b00); // write 0 - should not clear TO
    axi_read('h04, rd, rd_resp);
    if (rd[1] !== 1'b1) begin errors++; $display("FAIL: writing 0 to STATUS incorrectly cleared TO"); end

    // periodic mode reload: run long enough to guarantee at least two full
    // periods have elapsed (avoids a brittle cycle-exact wait), then check
    // TO fired and BUSY never dropped (periodic mode never stops on its own)
    axi_write('h04, 32'b10); // clear TO
    axi_write('h08, 32'd3);  // LOAD = 3
    axi_write('h00, 32'b011); // EN=1, MODE=1 (periodic)
    repeat (40) @(posedge clk);
    axi_read('h04, rd, rd_resp); // STATUS
    if (rd[0] !== 1'b1) begin errors++; $display("FAIL: periodic mode should still be BUSY, STATUS=%0h", rd); end
    if (rd[1] !== 1'b1) begin errors++; $display("FAIL: periodic mode should have set TO at least once, STATUS=%0h", rd); end
    axi_read('h0C, rd, rd_resp); // COUNT should be within the reload range
    if (rd > 32'd3) begin errors++; $display("FAIL: periodic COUNT=%0d out of range [0,3]", rd); end

    // LOAD write while BUSY -> wait state, staged (no corruption of the
    // in-flight count), and takes effect only at the next reload boundary.
    // Restart with a long period so the AXI round-trip latency of the
    // check itself can't accidentally straddle a reload boundary.
    axi_write('h04, 32'b10);  // clear TO
    axi_write('h08, 32'd50);  // LOAD = 50
    axi_write('h00, 32'b011); // EN=1, MODE=1 (periodic), restart from 50
    axi_write('h08, 32'd99);  // staged while BUSY: must not corrupt the in-flight count
    axi_read('h0C, rd, rd_resp);
    if (rd == 32'd99 || rd > 32'd50) begin errors++; $display("FAIL: LOAD write while BUSY corrupted in-flight COUNT=%0d", rd); end
    // run past a reload boundary (50 decrements + margin) and confirm the
    // staged value has now committed into LOAD
    repeat (70) @(posedge clk);
    axi_read('h08, rd, rd_resp); // LOAD
    if (rd !== 32'd99) begin errors++; $display("FAIL: staged LOAD did not commit at reload boundary, LOAD=%0d", rd); end

    // bad address -> SLVERR
    axi_read('h20, rd, rd_resp);
    if (rd_resp !== 2'b10) begin errors++; $display("FAIL: expected SLVERR on bad addr, got %0d", rd_resp); end

    if (errors == 0) $display("SANITY TB: ALL CHECKS PASSED");
    else $display("SANITY TB: %0d CHECKS FAILED", errors);
    $finish;
  end
endmodule

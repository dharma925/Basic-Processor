// AXI4-Lite interface used by the TB agent (driver + monitor) to connect
// to the axi4lite_apb_bridge DUT. Plain signals, no clocking block (see
// docs/README for why this project's testbenches avoid clocking blocks
// under Verilator's --timing class support).
//
// The drive_write()/drive_read()/reset_signals() tasks below live *inside*
// the interface (operating on its own signals directly) rather than being
// class methods/tasks that take a `virtual axi_lite_if` as a parameter.
// That's not just style: a task that takes a virtual interface parameter
// and both writes and reads different fields of it triggers a genuine
// a simulator bug ("Input combinational region did not converge",
// fires even if the task is never called) - see docs/README's Toolchain
// section for the bisected repro. Calling vif.drive_write(...) through the
// virtual handle instead of drive_write(vif, ...) avoids it entirely.
interface axi_lite_if #(parameter int ADDR_WIDTH = 32) (input logic aclk, input logic aresetn);
  logic [ADDR_WIDTH-1:0] awaddr;
  logic                  awvalid;
  logic                  awready;

  logic [31:0]           wdata;
  logic [3:0]            wstrb;
  logic                  wvalid;
  logic                  wready;

  logic [1:0]            bresp;
  logic                  bvalid;
  logic                  bready;

  logic [ADDR_WIDTH-1:0] araddr;
  logic                  arvalid;
  logic                  arready;

  logic [31:0]           rdata;
  logic [1:0]            rresp;
  logic                  rvalid;
  logic                  rready;

  task automatic reset_signals();
    awvalid = 0; awaddr = '0;
    wvalid  = 0; wdata  = '0; wstrb = 4'hF;
    bready  = 0;
    arvalid = 0; araddr = '0;
    rready  = 0;
  endtask

  // Timing convention (see docs/README): stimulus is applied a small delta
  // after the driving edge (@(posedge aclk); #1; sig = val;) so it never
  // races the DUT's own same-edge nonblocking updates; readiness is polled
  // at negedge (safely mid-cycle, strictly before the edge that actually
  // captures the handshake) rather than right after posedge, since by then
  // the DUT's state may have already advanced past the condition checked.
  task automatic drive_write(input logic [ADDR_WIDTH-1:0] addr, input logic [31:0] data,
                             input logic [3:0] strb, output logic [1:0] resp);
    @(posedge aclk); #1;
    awaddr = addr; awvalid = 1;
    wdata  = data; wstrb = strb; wvalid = 1;
    bready = 1;
    @(negedge aclk);
    while (!(awready && wready)) @(negedge aclk);
    @(posedge aclk); #1; // capturing edge
    awvalid = 0; wvalid = 0;
    @(negedge aclk);
    while (!bvalid) @(negedge aclk);
    resp = bresp;
    @(posedge aclk); #1;
    bready = 0;
  endtask

  task automatic drive_read(input logic [ADDR_WIDTH-1:0] addr, output logic [31:0] data,
                            output logic [1:0] resp);
    @(posedge aclk); #1;
    araddr = addr; arvalid = 1;
    rready = 1;
    @(negedge aclk);
    while (!arready) @(negedge aclk);
    @(posedge aclk); #1; // capturing edge
    arvalid = 0;
    @(negedge aclk);
    while (!rvalid) @(negedge aclk);
    data = rdata;
    resp = rresp;
    @(posedge aclk); #1;
    rready = 0;
  endtask

  // Passive monitor loop, for the same reason drive_write()/drive_read()
  // live here instead of in axi_monitor.sv: this class task, even though
  // it only *reads* vif fields, still takes a virtual-interface-typed
  // parameter in a task with real branching - which alone was enough to
  // reproduce the same "Input combinational region did not converge" bug.
  // Living inside the interface (own signals, no virtual-interface
  // parameter anywhere) sidesteps it.
  task automatic monitor_loop(tb_base_pkg::my_analysis_port #(axi_txn) ap);
    logic [ADDR_WIDTH-1:0] aw_addr, ar_addr;
    logic [31:0]           w_data;

    forever begin
      @(negedge aclk);
      if (!aresetn) continue;

      if (awvalid && awready) aw_addr = awaddr;
      if (wvalid && wready)   w_data  = wdata;
      if (bvalid && bready) begin
        axi_txn t = new("axi_write_obs");
        t.dir   = AXI_WRITE;
        t.addr  = aw_addr;
        t.wdata = w_data;
        t.resp  = bresp;
        ap.write(t);
      end

      if (arvalid && arready) ar_addr = araddr;
      if (rvalid && rready) begin
        axi_txn t = new("axi_read_obs");
        t.dir   = AXI_READ;
        t.addr  = ar_addr;
        t.rdata = rdata;
        t.resp  = rresp;
        ap.write(t);
      end
    end
  endtask
endinterface

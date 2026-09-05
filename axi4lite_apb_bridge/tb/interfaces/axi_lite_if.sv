// AXI4-Lite interface used by the TB agent (driver + monitor) to connect
// to the axi4lite_apb_bridge DUT. Plain signals, no clocking block (see
// docs/README for why this project's testbenches avoid clocking blocks
// under Verilator's --timing class support).
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
endinterface

// axi4lite_apb_bridge
//
// Single-outstanding AXI4-Lite slave <-> APB3 master bridge.
// Written from scratch for this project.
//
// Scope (deliberately bounded, see README):
//   * single-beat 32-bit reads/writes only, one outstanding AXI transaction
//     at a time (no pipelining of AW/W/AR ahead of a response)
//   * WSTRB is assumed to be 4'hF (full word) - narrow/partial transfers are
//     out of scope
//   * address decode: only offsets 0x00-0x10 (word-aligned, aliased on the
//     low ADDR_WIDTH bits) map to the attached APB peripheral; anything else
//     gets AXI SLVERR without ever asserting PSEL, i.e. an address decode
//     error, not a peripheral-reported error
//   * APB: standard SETUP/ACCESS handshake, waits for PREADY (supports
//     slave-inserted wait states), returns AXI SLVERR if PSLVERR is seen

module axi4lite_apb_bridge #(
  parameter int AXI_ADDR_WIDTH = 32,
  parameter int APB_ADDR_WIDTH = 5     // matches gpt_timer's decoded window
) (
  input  logic                        aclk,
  input  logic                        aresetn,

  // AXI4-Lite slave - write address channel
  input  logic [AXI_ADDR_WIDTH-1:0]   s_axi_awaddr,
  input  logic                        s_axi_awvalid,
  output logic                        s_axi_awready,

  // AXI4-Lite slave - write data channel
  input  logic [31:0]                 s_axi_wdata,
  input  logic [3:0]                  s_axi_wstrb,
  input  logic                        s_axi_wvalid,
  output logic                        s_axi_wready,

  // AXI4-Lite slave - write response channel
  output logic [1:0]                  s_axi_bresp,
  output logic                        s_axi_bvalid,
  input  logic                        s_axi_bready,

  // AXI4-Lite slave - read address channel
  input  logic [AXI_ADDR_WIDTH-1:0]   s_axi_araddr,
  input  logic                        s_axi_arvalid,
  output logic                        s_axi_arready,

  // AXI4-Lite slave - read data channel
  output logic [31:0]                 s_axi_rdata,
  output logic [1:0]                  s_axi_rresp,
  output logic                        s_axi_rvalid,
  input  logic                        s_axi_rready,

  // APB3 master
  output logic                        m_apb_psel,
  output logic                        m_apb_penable,
  output logic                        m_apb_pwrite,
  output logic [APB_ADDR_WIDTH-1:0]   m_apb_paddr,
  output logic [31:0]                 m_apb_pwdata,
  input  logic [31:0]                 m_apb_prdata,
  input  logic                        m_apb_pready,
  input  logic                        m_apb_pslverr
);

  localparam logic [1:0] RESP_OKAY   = 2'b00;
  localparam logic [1:0] RESP_SLVERR = 2'b10;

  // Address decode: valid offsets are 0x00, 0x04, 0x08, 0x0C, 0x10 and
  // nothing above the peripheral's own address window may be set.
  function automatic logic addr_in_range(input logic [AXI_ADDR_WIDTH-1:0] addr);
    addr_in_range = (addr[1:0] == 2'b00) &&
                    (addr[AXI_ADDR_WIDTH-1:APB_ADDR_WIDTH] == '0) &&
                    (addr[APB_ADDR_WIDTH-1:2] <= 3'd4);
  endfunction

  typedef enum logic [2:0] {
    S_IDLE,
    S_APB_SETUP,
    S_APB_ACCESS,
    S_WRITE_RESP,
    S_READ_RESP,
    S_DECERR_WRITE,
    S_DECERR_READ
  } state_e;

  state_e state_q, state_d;

  logic                      is_write_q;
  logic [AXI_ADDR_WIDTH-1:0] addr_q;
  logic [31:0]               wdata_q;
  logic [31:0]               rdata_q;
  logic [1:0]                resp_q;

  logic aw_have, w_have;

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      aw_have <= 1'b0;
      w_have  <= 1'b0;
    end else begin
      if (s_axi_awvalid && s_axi_awready) aw_have <= 1'b1;
      if (s_axi_wvalid  && s_axi_wready)  w_have  <= 1'b1;
      if (state_q == S_IDLE && state_d != S_IDLE) begin
        aw_have <= 1'b0;
        w_have  <= 1'b0;
      end
    end
  end

  wire write_req = (s_axi_awvalid || aw_have) && (s_axi_wvalid || w_have);
  wire read_req  = s_axi_arvalid;

  // accept AW/W as soon as both are (or become) available and we're about to
  // start a new transaction from IDLE
  assign s_axi_awready = (state_q == S_IDLE) && write_req && !aw_have;
  assign s_axi_wready  = (state_q == S_IDLE) && write_req && !w_have;
  assign s_axi_arready = (state_q == S_IDLE) && !write_req && read_req;

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      addr_q  <= '0;
      wdata_q <= '0;
      is_write_q <= 1'b0;
    end else if (state_q == S_IDLE && state_d != S_IDLE) begin
      is_write_q <= write_req;
      if (write_req) begin
        addr_q  <= s_axi_awvalid ? s_axi_awaddr : addr_q;
        wdata_q <= s_axi_wvalid  ? s_axi_wdata  : wdata_q;
      end else begin
        addr_q <= s_axi_araddr;
      end
    end
  end

  always_comb begin
    state_d = state_q;
    unique case (state_q)
      S_IDLE: begin
        if (write_req)
          state_d = addr_in_range(s_axi_awvalid ? s_axi_awaddr : addr_q) ? S_APB_SETUP : S_DECERR_WRITE;
        else if (read_req)
          state_d = addr_in_range(s_axi_araddr) ? S_APB_SETUP : S_DECERR_READ;
      end
      S_APB_SETUP:  state_d = S_APB_ACCESS;
      S_APB_ACCESS: if (m_apb_pready) state_d = is_write_q ? S_WRITE_RESP : S_READ_RESP;
      S_WRITE_RESP: if (s_axi_bready) state_d = S_IDLE;
      S_READ_RESP:  if (s_axi_rready) state_d = S_IDLE;
      S_DECERR_WRITE: if (s_axi_bready) state_d = S_IDLE;
      S_DECERR_READ:  if (s_axi_rready) state_d = S_IDLE;
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) state_q <= S_IDLE;
    else          state_q <= state_d;
  end

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      rdata_q <= '0;
      resp_q  <= RESP_OKAY;
    end else if (state_q == S_APB_ACCESS && m_apb_pready) begin
      rdata_q <= m_apb_prdata;
      resp_q  <= m_apb_pslverr ? RESP_SLVERR : RESP_OKAY;
    end else if (state_q == S_IDLE && (state_d == S_DECERR_WRITE || state_d == S_DECERR_READ)) begin
      // registered one cycle ahead of need, same as the APB completion case
      // above: BVALID/RVALID assert combinationally the instant the decerr
      // state is entered, so resp_q must already be correct by then.
      rdata_q <= 32'h0;
      resp_q  <= RESP_SLVERR;
    end
  end

  assign m_apb_psel    = (state_q == S_APB_SETUP) || (state_q == S_APB_ACCESS);
  assign m_apb_penable = (state_q == S_APB_ACCESS);
  assign m_apb_pwrite  = is_write_q;
  assign m_apb_paddr   = addr_q[APB_ADDR_WIDTH-1:0];
  assign m_apb_pwdata  = wdata_q;

  assign s_axi_bvalid = (state_q == S_WRITE_RESP) || (state_q == S_DECERR_WRITE);
  assign s_axi_bresp  = resp_q;

  assign s_axi_rvalid = (state_q == S_READ_RESP) || (state_q == S_DECERR_READ);
  assign s_axi_rdata  = rdata_q;
  assign s_axi_rresp  = resp_q;

endmodule

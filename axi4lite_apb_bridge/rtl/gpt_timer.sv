// GPT - Generic Programmable Timer
//
// Original, from-scratch APB3 peripheral: a single-channel down-counting
// timer/compare block. Register map (all registers are 32-bit, word aligned):
//
//   0x00 CTRL   RW          [0]EN [1]MODE [2]IE  (others reserved, read 0)
//   0x04 STATUS RO / W1C    [0]BUSY(RO)   [1]TO(W1C)
//   0x08 LOAD   RW          reload/compare value
//   0x0C COUNT  RO          live down-counter snapshot
//   0x10 ID     RO constant 0xC0FFEE01
//
// Behavior: writing CTRL.EN=1 while not already counting loads COUNT from
// LOAD and starts decrementing once per clock. On reaching 0, STATUS.TO is
// set; in periodic mode (CTRL.MODE=1) COUNT reloads from LOAD and keeps
// counting, otherwise BUSY clears and counting stops. A write to LOAD while
// BUSY=1 is staged and only takes effect at the next reload boundary so it
// cannot corrupt the in-flight count; that particular access (APB write to
// LOAD while BUSY=1) also inserts one PREADY wait state, giving the bridge
// a real wait-state case to handle.

module gpt_timer #(
  parameter int ADDR_WIDTH = 5    // byte address width covering this peripheral's window (0x00-0x1F)
) (
  input  logic                    pclk,
  input  logic                    presetn,

  input  logic                    psel,
  input  logic                    penable,
  input  logic                    pwrite,
  input  logic [ADDR_WIDTH-1:0]   paddr,
  input  logic [31:0]             pwdata,

  output logic [31:0]             prdata,
  output logic                    pready,
  output logic                    pslverr
);

  localparam logic [ADDR_WIDTH-1:0] ADDR_CTRL   = 'h00;
  localparam logic [ADDR_WIDTH-1:0] ADDR_STATUS = 'h04;
  localparam logic [ADDR_WIDTH-1:0] ADDR_LOAD   = 'h08;
  localparam logic [ADDR_WIDTH-1:0] ADDR_COUNT  = 'h0C;
  localparam logic [ADDR_WIDTH-1:0] ADDR_ID     = 'h10;
  localparam logic [31:0]           ID_VALUE    = 32'hC0FF_EE01;

  logic        ctrl_en, ctrl_mode, ctrl_ie;
  logic        status_busy, status_to;
  logic [31:0] load_q;
  logic [31:0] count_q;

  logic [31:0] load_pending;
  logic        load_pending_valid;

  logic access_phase;
  assign access_phase = psel && penable;

  logic is_load_write_busy;
  assign is_load_write_busy = psel && pwrite && (paddr == ADDR_LOAD) && status_busy;

  // one-wait-state generator: only used for LOAD writes while BUSY
  logic wait_done_q;
  always_ff @(posedge pclk or negedge presetn) begin
    if (!presetn)
      wait_done_q <= 1'b0;
    else if (!psel)
      wait_done_q <= 1'b0;
    else if (access_phase)
      wait_done_q <= 1'b1;
  end

  assign pready  = access_phase && (!is_load_write_busy || wait_done_q);
  assign pslverr = 1'b0; // this peripheral's own 5 registers always respond OKAY;
                          // out-of-range decode is handled by the bridge itself.

  wire write_fire = access_phase && pready && pwrite;
  wire read_fire  = access_phase && pready && !pwrite;

  always_ff @(posedge pclk or negedge presetn) begin
    if (!presetn) begin
      ctrl_en            <= 1'b0;
      ctrl_mode          <= 1'b0;
      ctrl_ie             <= 1'b0;
      status_busy        <= 1'b0;
      status_to          <= 1'b0;
      load_q             <= 32'h0;
      count_q            <= 32'h0;
      load_pending       <= 32'h0;
      load_pending_valid <= 1'b0;
    end else begin
      // ---- counting / reload / completion ----
      if (status_busy) begin
        if (count_q == 32'h0) begin
          status_to <= 1'b1;
          if (ctrl_mode) begin
            if (load_pending_valid) begin
              load_q             <= load_pending;
              count_q            <= load_pending;
              load_pending_valid <= 1'b0;
            end else begin
              count_q <= load_q;
            end
          end else begin
            status_busy <= 1'b0;
          end
        end else begin
          count_q <= count_q - 32'h1;
        end
      end

      // ---- register writes ----
      if (write_fire) begin
        case (paddr)
          ADDR_CTRL: begin
            ctrl_en   <= pwdata[0];
            ctrl_mode <= pwdata[1];
            ctrl_ie   <= pwdata[2];
            if (pwdata[0] && !status_busy) begin
              count_q     <= load_q;
              status_busy <= 1'b1;
            end
          end
          ADDR_STATUS: begin
            if (pwdata[1]) status_to <= 1'b0; // W1C; writing 0 has no effect
          end
          ADDR_LOAD: begin
            if (status_busy) begin
              load_pending       <= pwdata;
              load_pending_valid <= 1'b1;
            end else begin
              load_q <= pwdata;
            end
          end
          default: ; // COUNT, ID are read-only: writes ignored
        endcase
      end
    end
  end

  always_comb begin
    case (paddr)
      ADDR_CTRL:   prdata = {29'h0, ctrl_ie, ctrl_mode, ctrl_en};
      ADDR_STATUS: prdata = {30'h0, status_to, status_busy};
      ADDR_LOAD:   prdata = load_q;
      ADDR_COUNT:  prdata = count_q;
      ADDR_ID:     prdata = ID_VALUE;
      default:     prdata = 32'h0;
    endcase
  end

endmodule

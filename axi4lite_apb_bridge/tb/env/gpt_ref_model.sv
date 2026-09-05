// Behavioral reference model of the GPT timer's register semantics,
// written independently from rtl/gpt_timer.sv (from the register-map spec,
// not by copying the RTL) so the scoreboard checks real intended behavior
// rather than just RTL-vs-RTL data echo.
//
// Cycle-accuracy comes from apb_if's own run_ref_model() task calling
// tick() once per pclk edge with live bus values, exactly like a
// synchronous shadow register file - see apb_if.sv's header for why that
// loop lives in the interface rather than as a method here that would take
// `virtual apb_if` as a parameter (a simulator bug, not a style choice).
class gpt_ref_model;
  localparam logic [4:0] ADDR_CTRL   = 5'h00;
  localparam logic [4:0] ADDR_STATUS = 5'h04;
  localparam logic [4:0] ADDR_LOAD   = 5'h08;
  localparam logic [4:0] ADDR_COUNT  = 5'h0C;
  localparam logic [4:0] ADDR_ID     = 5'h10;
  localparam logic [31:0] ID_VALUE   = 32'hC0FF_EE01;

  bit          ctrl_en, ctrl_mode, ctrl_ie;
  bit          status_busy, status_to;
  logic [31:0] load_q, count_q;
  logic [31:0] load_pending;
  bit          load_pending_valid;

  function new();
    reset();
  endfunction

  function void reset();
    ctrl_en = 0; ctrl_mode = 0; ctrl_ie = 0;
    status_busy = 0; status_to = 0;
    load_q = 0; count_q = 0;
    load_pending = 0; load_pending_valid = 0;
  endfunction

  function logic [31:0] read(logic [4:0] addr);
    case (addr)
      ADDR_CTRL:   read = {29'h0, ctrl_ie, ctrl_mode, ctrl_en};
      ADDR_STATUS: read = {30'h0, status_to, status_busy};
      ADDR_LOAD:   read = load_q;
      ADDR_COUNT:  read = count_q;
      ADDR_ID:     read = ID_VALUE;
      default:     read = 32'h0;
    endcase
  endfunction

  // advance one clock; write_fire/waddr/wdata describe a write completing
  // on this same edge (mirrors the priority order in the spec: counting
  // advances first, then the write applies).
  function automatic void tick(bit write_fire, logic [4:0] waddr, logic [31:0] wdata);
    if (status_busy) begin
      if (count_q == 32'h0) begin
        status_to = 1;
        if (ctrl_mode) begin
          if (load_pending_valid) begin
            load_q = load_pending;
            count_q = load_pending;
            load_pending_valid = 0;
          end else begin
            count_q = load_q;
          end
        end else begin
          status_busy = 0;
        end
      end else begin
        count_q = count_q - 32'h1;
      end
    end

    if (write_fire) begin
      case (waddr)
        ADDR_CTRL: begin
          ctrl_en = wdata[0]; ctrl_mode = wdata[1]; ctrl_ie = wdata[2];
          if (wdata[0] && !status_busy) begin
            count_q = load_q;
            status_busy = 1;
          end
        end
        ADDR_STATUS: if (wdata[1]) status_to = 0;
        ADDR_LOAD: begin
          if (status_busy) begin
            load_pending = wdata;
            load_pending_valid = 1;
          end else begin
            load_q = wdata;
          end
        end
        default: ; // COUNT, ID read-only
      endcase
    end
  endfunction

endclass

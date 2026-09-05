// Functional coverage, sampled from the same apb_txn/axi_txn analysis
// ports the scoreboard listens to.
//
// The SystemVerilog covergroup/coverpoint constructs are not implemented by
// our simulator (rejected with "Unsupported: covergroup" even outside a
// class), so this is a small hand-rolled bins model instead of a real
// covergroup - the same "UVM-styled, not the real thing, because the
// open-source tool can't do it" tradeoff as tb_base_pkg (see its header
// and the README).
class gpt_coverage extends tb_base_pkg::my_component;
  bit hit_reg_write[5];
  bit hit_reg_read[5];
  bit hit_mode_one_shot, hit_mode_periodic;
  bit hit_to_set, hit_to_cleared;
  bit hit_wait_state, hit_no_wait;
  bit hit_resp_okay, hit_resp_slverr;

  function new(string name = "gpt_coverage");
    super.new(name);
  endfunction

  function automatic int reg_index(logic [4:0] addr);
    case (addr)
      5'h00:   return 0;
      5'h04:   return 1;
      5'h08:   return 2;
      5'h0C:   return 3;
      5'h10:   return 4;
      default: return -1;
    endcase
  endfunction

  function automatic void sample_apb(apb_txn t);
    int idx;
    idx = reg_index(t.addr[4:0]);
    if (idx >= 0) begin
      if (t.write) hit_reg_write[idx] = 1;
      else         hit_reg_read[idx]  = 1;
    end
    if (t.wait_cycles > 0) hit_wait_state = 1;
    else                   hit_no_wait    = 1;
    if (t.addr[4:0] == 5'h00 && t.write) begin
      if (t.data[1]) hit_mode_periodic = 1;
      else           hit_mode_one_shot = 1;
    end
    if (t.addr[4:0] == 5'h04) begin
      if (t.data[1]) hit_to_set     = 1;
      else           hit_to_cleared = 1;
    end
  endfunction

  function automatic void sample_axi(axi_txn t);
    if (t.resp == 2'b00)      hit_resp_okay   = 1;
    else if (t.resp == 2'b10) hit_resp_slverr = 1;
  endfunction

  function automatic real get_coverage();
    int total, hit;
    total = 0; hit = 0;
    for (int i = 0; i < 5; i++) begin
      total += 2;
      hit   += hit_reg_write[i] + hit_reg_read[i];
    end
    total += 6;
    hit += hit_mode_one_shot + hit_mode_periodic + hit_to_set + hit_to_cleared
         + hit_wait_state + hit_no_wait;
    total += 2;
    hit += hit_resp_okay + hit_resp_slverr;
    return 100.0 * hit / total;
  endfunction

  function automatic void report();
    $display("COVERAGE detail:");
    $display("  reg write hits: CTRL=%0d STATUS=%0d LOAD=%0d COUNT=%0d ID=%0d",
              hit_reg_write[0], hit_reg_write[1], hit_reg_write[2], hit_reg_write[3], hit_reg_write[4]);
    $display("  reg read  hits: CTRL=%0d STATUS=%0d LOAD=%0d COUNT=%0d ID=%0d",
              hit_reg_read[0], hit_reg_read[1], hit_reg_read[2], hit_reg_read[3], hit_reg_read[4]);
    $display("  mode: one_shot=%0d periodic=%0d | TO: set=%0d cleared=%0d | wait: seen=%0d none=%0d | resp: okay=%0d slverr=%0d",
              hit_mode_one_shot, hit_mode_periodic, hit_to_set, hit_to_cleared,
              hit_wait_state, hit_no_wait, hit_resp_okay, hit_resp_slverr);
  endfunction
endclass

class cov_apb_adapter extends tb_base_pkg::my_subscriber #(apb_txn);
  gpt_coverage cov;
  function new(gpt_coverage cov);
    this.cov = cov;
  endfunction
  function void write(apb_txn t);
    cov.sample_apb(t);
  endfunction
endclass

class cov_axi_adapter extends tb_base_pkg::my_subscriber #(axi_txn);
  gpt_coverage cov;
  function new(gpt_coverage cov);
    this.cov = cov;
  endfunction
  function void write(axi_txn t);
    cov.sample_axi(t);
  endfunction
endclass

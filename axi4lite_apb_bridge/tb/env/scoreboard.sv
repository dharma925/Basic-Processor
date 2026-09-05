// Scoreboard: checks AXI write -> correct APB write, AXI read -> correct
// APB read -> correct data returned on AXI, and additionally checks the
// APB-observed data against gpt_ref_model (a behavioral model of the GPT's
// register semantics, not just an RTL-vs-RTL echo check).
//
// SV classes don't support extending two parameterized base classes, so the
// scoreboard exposes plain write_axi()/write_apb() methods and two tiny
// adapter objects (below) forward each analysis port's write() into them.
class scoreboard extends tb_base_pkg::my_component;
  gpt_ref_model ref_model;
  apb_txn       apb_pending[$];
  int           num_checked;
  int           num_errors;

  function new(string name, gpt_ref_model ref_model);
    super.new(name);
    this.ref_model = ref_model;
    num_checked = 0;
    num_errors  = 0;
  endfunction

  function automatic bit addr_in_range(logic [31:0] addr);
    addr_in_range = (addr[1:0] == 2'b00) && (addr[31:5] == '0) && (addr[4:2] <= 3'd4);
  endfunction

  function automatic void write_apb(apb_txn t);
    logic [31:0] expected;
    // captured "now" (before this transaction's own tick effect lands),
    // matching exactly what PRDATA held during this APB access phase.
    expected = ref_model.read(t.addr[4:0]);
    if (!t.write && t.data !== expected) begin
      num_errors++;
      $display("SCOREBOARD: APB read mismatch addr=%0h got=%0h expected(ref_model)=%0h",
             t.addr, t.data, expected);
    end
    apb_pending.push_back(t);
  endfunction

  function automatic void write_axi(axi_txn t);
    num_checked++;
    if (addr_in_range(t.addr)) begin
      apb_txn apb_t;
      if (apb_pending.size() == 0) begin
        num_errors++;
        $display("SCOREBOARD: AXI %s addr=%0h had no matching APB transaction",
               t.dir.name(), t.addr);
        return;
      end
      apb_t = apb_pending.pop_front();
      if (apb_t.addr !== t.addr[4:0]) begin
        num_errors++;
        $display("SCOREBOARD: address mismatch AXI=%0h APB=%0h", t.addr, apb_t.addr);
      end
      if (t.dir == AXI_WRITE) begin
        if (!apb_t.write) begin
          num_errors++;
          $display("SCOREBOARD: AXI WRITE addr=%0h paired with an APB READ", t.addr);
        end else if (apb_t.data !== t.wdata) begin
          num_errors++;
          $display("SCOREBOARD: AXI WRITE data=%0h did not reach APB (got %0h)", t.wdata, apb_t.data);
        end
        if (t.resp !== 2'b00) begin
          num_errors++;
          $display("SCOREBOARD: in-range AXI WRITE got non-OKAY resp=%0d", t.resp);
        end
      end else begin // AXI_READ
        if (apb_t.write) begin
          num_errors++;
          $display("SCOREBOARD: AXI READ addr=%0h paired with an APB WRITE", t.addr);
        end else if (apb_t.data !== t.rdata) begin
          num_errors++;
          $display("SCOREBOARD: APB read data=%0h did not reach AXI (got %0h)", apb_t.data, t.rdata);
        end
        if (t.resp !== 2'b00) begin
          num_errors++;
          $display("SCOREBOARD: in-range AXI READ got non-OKAY resp=%0d", t.resp);
        end
      end
    end else begin
      if (t.resp !== 2'b10) begin
        num_errors++;
        $display("SCOREBOARD: out-of-range AXI %s addr=%0h expected SLVERR, got resp=%0d",
               t.dir.name(), t.addr, t.resp);
      end
      if (apb_pending.size() != 0) begin
        num_errors++;
        $display("SCOREBOARD: out-of-range addr=%0h unexpectedly produced an APB transaction", t.addr);
      end
    end
  endfunction

  function automatic void report();
    $display("[%s] checked=%0d errors=%0d", name, num_checked, num_errors);
  endfunction
endclass

class axi_sub_adapter extends tb_base_pkg::my_subscriber #(axi_txn);
  scoreboard sb;
  function new(scoreboard sb);
    this.sb = sb;
  endfunction
  function void write(axi_txn t);
    sb.write_axi(t);
  endfunction
endclass

class apb_sub_adapter extends tb_base_pkg::my_subscriber #(apb_txn);
  scoreboard sb;
  function new(scoreboard sb);
    this.sb = sb;
  endfunction
  function void write(apb_txn t);
    sb.write_apb(t);
  endfunction
endclass

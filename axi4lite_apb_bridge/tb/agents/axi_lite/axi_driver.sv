// AXI4-Lite driver: pulls axi_txn items from its sequencer and drives them
// onto s_axi_* via axi_lite_if's own drive_write()/drive_read() tasks, one
// item at a time (this project's bridge is single-outstanding, so the
// driver never overlaps transfers).
//
// The actual pin-wiggling lives inside axi_lite_if itself rather than
// here, so this class only ever calls through the virtual handle
// (vif.drive_write(...)) instead of taking `virtual axi_lite_if` as a task
// parameter - see axi_lite_if.sv's header for the Verilator bug that
// distinction avoids.
class axi_driver extends tb_base_pkg::my_component;
  virtual axi_lite_if vif;
  tb_base_pkg::my_sequencer #(axi_txn) sqr;

  function new(string name, virtual axi_lite_if vif, tb_base_pkg::my_sequencer #(axi_txn) sqr);
    super.new(name);
    this.vif = vif;
    this.sqr = sqr;
  endfunction

  task automatic run();
    axi_txn item;
    vif.reset_signals();
    forever begin
      sqr.get_next_item(item);
      if (item.dir == AXI_WRITE) vif.drive_write(item.addr, item.wdata, item.wstrb, item.resp);
      else                       vif.drive_read(item.addr, item.rdata, item.resp);
      sqr.item_done();
    end
  endtask
endclass

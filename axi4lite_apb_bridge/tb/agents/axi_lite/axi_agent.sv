// AXI4-Lite agent: bundles sequencer + driver + monitor, mirroring a
// uvm_agent. Active (drives the bus) since this is the DUT's AXI slave
// interface and something has to stimulate it.
class axi_agent extends tb_base_pkg::my_component;
  tb_base_pkg::my_sequencer #(axi_txn) sqr;
  axi_driver  drv;
  axi_monitor mon;

  function new(string name, virtual axi_lite_if vif);
    super.new(name);
    sqr = new({name, ".sqr"});
    drv = new({name, ".drv"}, vif, sqr);
    mon = new({name, ".mon"}, vif);
  endfunction

  task automatic run();
    fork
      drv.run();
      mon.run();
    join_none
  endtask
endclass

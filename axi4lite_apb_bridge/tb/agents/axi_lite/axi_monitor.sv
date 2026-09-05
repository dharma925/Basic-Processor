// AXI4-Lite monitor: thin wrapper around axi_lite_if's own monitor_loop()
// (see that file's header for why the actual sampling logic lives inside
// the interface rather than here).
class axi_monitor extends tb_base_pkg::my_component;
  virtual axi_lite_if vif;
  tb_base_pkg::my_analysis_port #(axi_txn) ap;

  function new(string name, virtual axi_lite_if vif);
    super.new(name);
    this.vif = vif;
    ap = new();
  endfunction

  task automatic run();
    vif.monitor_loop(ap);
  endtask
endclass

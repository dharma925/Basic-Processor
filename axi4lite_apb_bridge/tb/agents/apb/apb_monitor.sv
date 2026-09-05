// APB monitor: thin wrapper around apb_if's own monitor_loop() (see that
// file's header for why the actual sampling logic lives inside the
// interface rather than here). P0 scope calls for a monitor only on this
// bus - the bridge is the only APB master, the GPT timer the only slave.
class apb_monitor extends tb_base_pkg::my_component;
  virtual apb_if vif;
  tb_base_pkg::my_analysis_port #(apb_txn) ap;

  function new(string name, virtual apb_if vif);
    super.new(name);
    this.vif = vif;
    ap = new();
  endfunction

  task automatic run();
    vif.monitor_loop(ap);
  endtask
endclass

// Top-level verification environment: AXI agent + APB monitor + reference
// model + scoreboard + coverage, wired together like a uvm_env's connect
// phase would.
class gpt_env extends tb_base_pkg::my_component;
  axi_agent     axi_agt;
  apb_monitor   apb_mon;
  gpt_ref_model ref_model;
  scoreboard    sb;
  gpt_coverage  cov;

  axi_sub_adapter axi_sb_adapter;
  apb_sub_adapter apb_sb_adapter;
  cov_axi_adapter axi_cov_adapter;
  cov_apb_adapter apb_cov_adapter;

  function new(string name, virtual axi_lite_if axi_vif, virtual apb_if apb_vif);
    super.new(name);
    axi_agt   = new({name, ".axi_agt"}, axi_vif);
    apb_mon   = new({name, ".apb_mon"}, apb_vif);
    ref_model = new();
    sb        = new({name, ".sb"}, ref_model);
    cov       = new({name, ".cov"});

    axi_sb_adapter  = new(sb);
    apb_sb_adapter  = new(sb);
    axi_cov_adapter = new(cov);
    apb_cov_adapter = new(cov);

    axi_agt.mon.ap.connect(axi_sb_adapter);
    axi_agt.mon.ap.connect(axi_cov_adapter);
    apb_mon.ap.connect(apb_sb_adapter);
    apb_mon.ap.connect(apb_cov_adapter);
  endfunction

  task automatic run(virtual apb_if apb_vif);
    fork
      axi_agt.run();
      apb_mon.run();
      apb_vif.run_ref_model(ref_model);
    join_none
  endtask
endclass

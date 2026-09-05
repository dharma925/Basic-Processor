// AXI4-Lite transaction: the sequence item driven onto s_axi_* by axi_driver
// and reconstructed by axi_monitor. Response fields are filled in by the
// driver (from the DUT) so a sequence can inspect them right after
// finish_item() returns, since driver and sequence share the same handle.
typedef enum { AXI_READ, AXI_WRITE } axi_dir_e;

class axi_txn extends tb_base_pkg::my_sequence_item;
  axi_dir_e         dir;
  logic [31:0]      addr;
  logic [31:0]      wdata;   // valid for WRITE
  logic [3:0]       wstrb;   // valid for WRITE, default 4'hF

  // filled in by the driver after the transfer completes
  logic [31:0]      rdata;   // valid for READ
  logic [1:0]       resp;    // AXI resp: 2'b00 OKAY, 2'b10 SLVERR

  function new(string name = "axi_txn");
    super.new(name);
    wstrb = 4'hF;
  endfunction

  function string convert2string();
    if (dir == AXI_WRITE)
      return $sformatf("AXI_WRITE addr=%08h data=%08h resp=%0d", addr, wdata, resp);
    else
      return $sformatf("AXI_READ  addr=%08h data=%08h resp=%0d", addr, rdata, resp);
  endfunction
endclass

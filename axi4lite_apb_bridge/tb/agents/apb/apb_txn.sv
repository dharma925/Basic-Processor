// APB transfer as observed by apb_monitor: one entry per completed
// PSEL&&PENABLE&&PREADY handshake.
class apb_txn extends tb_base_pkg::my_sequence_item;
  bit          write;
  logic [31:0] addr;
  logic [31:0] data;   // pwdata for a write, prdata for a read
  bit          slverr;
  int          wait_cycles; // number of PREADY-deasserted cycles observed

  function new(string name = "apb_txn");
    super.new(name);
  endfunction

  function string convert2string();
    return $sformatf("APB_%s addr=%08h data=%08h slverr=%0b wait=%0d",
                      write ? "WRITE" : "READ", addr, data, slverr, wait_cycles);
  endfunction
endclass

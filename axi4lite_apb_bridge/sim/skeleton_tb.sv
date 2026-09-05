`timescale 1ns/1ps
// Stage-2 sanity check: does the hand-written UVM-styled base package
// compile and run a minimal sequence/sequencer/item handshake, with zero
// DUT or agent code involved yet.
import tb_base_pkg::*;

class dummy_item extends my_sequence_item;
  int value;
  function new(string name = "dummy_item");
    super.new(name);
  endfunction
endclass

class dummy_sequence extends my_sequence #(dummy_item);
  task body();
    dummy_item it;
    it = new("it0");
    it.value = 42;
    start_item(it);
    finish_item(it);
  endtask
endclass

module skeleton_tb;
  my_sequencer #(dummy_item) sqr;
  dummy_sequence seq;
  dummy_item got;

  initial begin
    sqr = new("sqr0");
    seq = new();
    fork
      seq.start(sqr);
    join_none
    sqr.get_next_item(got);
    $display("[%0t] skeleton_tb: got item '%s' value=%0d", $time, got.convert2string(), got.value);
    if (got.value == 42) $display("SKELETON TB: PASSED");
    else $display("SKELETON TB: FAILED");
    $finish;
  end
endmodule

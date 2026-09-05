// tb_base_pkg
//
// Hand-written, UVM-*styled* base classes: our own uvm_component /
// uvm_driver / uvm_monitor / uvm_scoreboard / uvm_sequencer /
// uvm_sequence_item / uvm_analysis_port stand-ins, mirroring the real UVM
// class hierarchy and TLM-style analysis ports closely enough that porting
// this environment to a real uvm_pkg later is mostly a rename.
//
// See docs/README.md "Toolchain" section for why: the real Accellera
// uvm_pkg does not compile under Verilator 5.020 (a documented nested
// parameterized-class limitation), so this project implements the
// methodology directly on Verilator's native SystemVerilog class support.
package tb_base_pkg;

  // ---- sequence_item -----------------------------------------------------
  virtual class my_sequence_item;
    string name;
    function new(string name = "my_sequence_item");
      this.name = name;
    endfunction
    virtual function string convert2string();
      return name;
    endfunction
  endclass

  // ---- analysis port (TLM-style broadcast) -------------------------------
  virtual class my_subscriber #(type T = my_sequence_item);
    pure virtual function void write(T t);
  endclass

  class my_analysis_port #(type T = my_sequence_item);
    protected my_subscriber#(T) subs[$];
    function void connect(my_subscriber#(T) sub);
      subs.push_back(sub);
    endfunction
    function void write(T t);
      foreach (subs[i]) subs[i].write(t);
    endfunction
  endclass

  // ---- component base -----------------------------------------------------
  virtual class my_component;
    string name;
    function new(string name = "my_component");
      this.name = name;
    endfunction
  endclass

  // ---- sequencer -----------------------------------------------------------
  // Mailbox-based req/rsp hand-off, mirroring uvm_sequencer's
  // get_next_item()/item_done() driver-facing API closely enough to be
  // recognizable.
  class my_sequencer #(type T = my_sequence_item) extends my_component;
    mailbox #(T) req_mbx;
    event        item_done_ev;
    function new(string name = "my_sequencer");
      super.new(name);
      req_mbx = new();
    endfunction
    task automatic get_next_item(output T item);
      req_mbx.get(item);
    endtask
    function void item_done();
      -> item_done_ev;
    endfunction
  endclass

  // ---- sequence ------------------------------------------------------------
  virtual class my_sequence #(type T = my_sequence_item);
    my_sequencer#(T) sqr;
    function void set_sequencer(my_sequencer#(T) sqr);
      this.sqr = sqr;
    endfunction
    task automatic start_item(T item);
      sqr.req_mbx.put(item);
    endtask
    task automatic finish_item(T item);
      // blocks until the driver calls sqr.item_done() for this item, so
      // the item's response fields are guaranteed populated once this
      // returns (single-outstanding: only one item is ever in flight).
      @(sqr.item_done_ev);
    endtask
    pure virtual task body();
    task automatic start(my_sequencer#(T) sqr);
      set_sequencer(sqr);
      body();
    endtask
  endclass

  // ---- lightweight pass/fail tally, used by directed sequences for
  //      explicit self-checks in addition to whatever the scoreboard
  //      already verifies automatically ---------------------------------
  class test_report;
    static int num_checks = 0;
    static int num_fails  = 0;
    static function void check(bit cond, string msg);
      num_checks++;
      if (!cond) begin
        num_fails++;
        // $display, not $error: our simulator treats $error as fatal
        // (aborts immediately), which would stop a directed sequence
        // after its first failed check instead of tallying all of them.
        $display("CHECK FAILED: %s", msg);
      end
    endfunction
    static function void summary();
      $display("TEST_REPORT: checks=%0d fails=%0d", num_checks, num_fails);
    endfunction
  endclass

endpackage

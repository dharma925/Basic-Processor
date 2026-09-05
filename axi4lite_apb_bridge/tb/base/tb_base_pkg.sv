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
    function new(string name = "my_sequencer");
      super.new(name);
      req_mbx = new();
    endfunction
    task automatic get_next_item(output T item);
      req_mbx.get(item);
    endtask
    function void item_done();
      // present for API symmetry with uvm_sequencer; this environment's
      // driver reports completion by filling in the item's response
      // fields directly (the sequence already holds the same handle).
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
      // synchronous driver: by the time the driver moves on to the next
      // mailbox item, this one's response fields are already populated.
    endtask
    pure virtual task body();
    task automatic start(my_sequencer#(T) sqr);
      set_sequencer(sqr);
      body();
    endtask
  endclass

endpackage

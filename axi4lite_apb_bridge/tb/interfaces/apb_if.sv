// APB3 interface. The TB only ever monitors this bus (the bridge itself is
// the APB master, the GPT timer is the APB slave); the assertions below
// check protocol legality of whatever is driving/responding on it.
interface apb_if #(parameter int ADDR_WIDTH = 5) (input logic pclk, input logic presetn);
  logic                  psel;
  logic                  penable;
  logic                  pwrite;
  logic [ADDR_WIDTH-1:0] paddr;
  logic [31:0]           pwdata;
  logic [31:0]           prdata;
  logic                  pready;
  logic                  pslverr;

  // ---------------------------------------------------------------------
  // SVA: APB protocol legality checks (P0 requirement: 4-6 assertions)
  // ---------------------------------------------------------------------

  // 1) PENABLE may only be asserted while PSEL is asserted.
  property p_penable_needs_psel;
    @(posedge pclk) disable iff (!presetn)
    penable |-> psel;
  endproperty
  a_penable_needs_psel: assert property (p_penable_needs_psel)
    else $error("APB: PENABLE asserted without PSEL");

  // 2) SETUP->ACCESS shape: PSEL asserted with PENABLE low must be followed
  //    by PENABLE high on the very next cycle (single-cycle SETUP phase).
  property p_setup_then_access;
    @(posedge pclk) disable iff (!presetn)
    (psel && !penable) |=> (psel && penable);
  endproperty
  a_setup_then_access: assert property (p_setup_then_access)
    else $error("APB: SETUP phase not followed by ACCESS phase");

  // 3) Once in the ACCESS phase without PREADY, PSEL/PENABLE/PADDR/PWRITE
  //    must hold (no new transaction may start before the previous one
  //    completes - this project's single-outstanding scope).
  property p_access_holds_until_ready;
    @(posedge pclk) disable iff (!presetn)
    (psel && penable && !pready) |=>
      (psel && penable && $stable(paddr) && $stable(pwrite));
  endproperty
  a_access_holds_until_ready: assert property (p_access_holds_until_ready)
    else $error("APB: transfer address/control changed while waiting for PREADY");

  // 4) PWDATA must be stable through a wait-stated write access.
  property p_wdata_stable_during_wait;
    @(posedge pclk) disable iff (!presetn)
    (psel && penable && pwrite && !pready) |=> $stable(pwdata);
  endproperty
  a_wdata_stable_during_wait: assert property (p_wdata_stable_during_wait)
    else $error("APB: PWDATA changed while waiting for PREADY on a write");

  // 5) After a completed transfer (PSEL&&PENABLE&&PREADY), PENABLE must drop
  //    the following cycle (back to IDLE or a fresh SETUP), never stay
  //    asserted through into a second ACCESS cycle for the same transfer.
  property p_penable_drops_after_ready;
    @(posedge pclk) disable iff (!presetn)
    (psel && penable && pready) |=> !penable;
  endproperty
  a_penable_drops_after_ready: assert property (p_penable_drops_after_ready)
    else $error("APB: PENABLE stayed asserted after a completed transfer");

  // 6) No X propagation on control signals during any active access phase.
  property p_no_x_on_control;
    @(posedge pclk) disable iff (!presetn)
    (psel && penable) |-> (!$isunknown(pwrite) && !$isunknown(paddr) && !$isunknown(pready));
  endproperty
  a_no_x_on_control: assert property (p_no_x_on_control)
    else $error("APB: X on control signals during an active access phase");

endinterface

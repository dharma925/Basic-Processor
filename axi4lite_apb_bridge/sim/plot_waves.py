#!/usr/bin/env python3
"""Render a labeled digital-timing PNG from waves.vcd, showing one AXI
write and one AXI read crossing over to APB. Used because this environment
has no X server for an interactive gtkwave screenshot."""
import sys
from vcdvcd import VCDVCD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vcd_path = sys.argv[1] if len(sys.argv) > 1 else "waves.vcd"
out_path = sys.argv[2] if len(sys.argv) > 2 else "waveform.png"

vcd = VCDVCD(vcd_path)

def sig(name):
    for k in vcd.signals:
        if k.endswith(name):
            return vcd[k]
    raise KeyError(name)

names = [
    ("clk", "tb_top.clk"),
    ("AXI awvalid", "axi_vif.awvalid"),
    ("AXI awready", "axi_vif.awready"),
    ("AXI wvalid", "axi_vif.wvalid"),
    ("AXI wready", "axi_vif.wready"),
    ("AXI bvalid", "axi_vif.bvalid"),
    ("AXI bready", "axi_vif.bready"),
    ("APB psel", "apb_vif.psel"),
    ("APB penable", "apb_vif.penable"),
    ("APB pwrite", "apb_vif.pwrite"),
    ("APB pready", "apb_vif.pready"),
    ("AXI arvalid", "axi_vif.arvalid"),
    ("AXI arready", "axi_vif.arready"),
    ("AXI rvalid", "axi_vif.rvalid"),
    ("AXI rready", "axi_vif.rready"),
]

data = {}
tmax = 0
for label, path in names:
    s = sig(path)
    tv = [(int(t), v) for t, v in s.tv]
    data[label] = tv
    tmax = max(tmax, tv[-1][0])

def to_steps(tv, tmax, period=10000):
    ts, vs = [], []
    for i, (t, v) in enumerate(tv):
        val = 1 if v == "1" else 0
        end = tv[i + 1][0] if i + 1 < len(tv) else tmax
        ts += [t, end]
        vs += [val, val]
    return ts, vs

fig, axes = plt.subplots(len(names), 1, figsize=(13, 9), sharex=True)
fig.suptitle("AXI4-Lite write+read crossing to APB (seq_ctrl_load_rw)", fontsize=13, y=0.995)

for ax, (label, _) in zip(axes, names):
    ts, vs = to_steps(data[label], tmax)
    ts_ns = [t / 1000.0 for t in ts]
    ax.plot(ts_ns, vs, drawstyle="steps-post", linewidth=1.6, color="#1a6fb0")
    ax.fill_between(ts_ns, vs, step="post", alpha=0.15, color="#1a6fb0")
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 1])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

axes[-1].set_xlabel("time (ns)")
axes[-1].set_xlim(30, 190)

# annotate the AXI write -> APB write crossing and the AXI read -> APB read crossing
axes[0].annotate("AXI write accepted\n(AWVALID&AWREADY, WVALID&WREADY)",
                  xy=(45, 1.4), xycoords=("data", "axes fraction"),
                  fontsize=8, ha="left", color="#b03a1a")
axes[7].annotate("bridge issues the matching\nAPB write (PSEL/PENABLE/PWRITE)",
                  xy=(65, 1.4), xycoords=("data", "axes fraction"),
                  fontsize=8, ha="left", color="#b03a1a")
axes[5].annotate("BVALID/BREADY completes\nthe AXI write response",
                  xy=(85, 1.4), xycoords=("data", "axes fraction"),
                  fontsize=8, ha="left", color="#b03a1a")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(out_path, dpi=150)
print(f"wrote {out_path}")

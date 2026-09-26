# -*- coding: utf-8 -*-
"""Paper-grade figure for P1 edge-roughness scan (hex_ribbon_dev).

Reads  data/hex_rib_edge_rough_er.json  (4 A0 x 2 depth x 6 p x 20 conf = 960 devices)
Output: hex_rib_edge_rough_er.png  (3 panels, English labels)

Panels:
 (a) per-config G vs p   (window A0 in color, depth marker; A0=0.1 control gray)
 (b) per-config P vs p
 (c) 1 - G_avg vs p, log y, six window combos + 0.5% Anderson reference line
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "hex_rib_edge_rough_er.json")
OUT  = os.path.join(HERE, "hex_rib_edge_rough_er.png")

d = json.load(open(DATA))
r = d["results"]

A0_WIN  = ["0.115", "0.118", "0.12"]
A0_CTRL = "0.1"
PS  = ["0", "0.05", "0.1", "0.2", "0.3", "0.5"]
PX = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
DEPS = ["1", "2"]

COL = {"0.115": "#1f77b4", "0.118": "#d62728", "0.12": "#2ca02c"}
MARK = {"1": "o", "2": "s"}

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

# ---- (a) G_conf scatter ----
ax = axes[0]
for a0 in A0_WIN:
    for dep in DEPS:
        xs, ys = [], []
        for p in PS:
            g = r[a0][dep][p]
            xs += [PX[PS.index(p)]] * len(g["G_conf"])
            ys += g["G_conf"]
        ax.scatter(xs, ys, s=14, marker=MARK[dep], facecolor="none",
                   edgecolor=COL[a0], linewidth=0.9,
                   label=f"$A_0$={a0}, depth {dep}", zorder=3)
xs, ys = [], []
for p in PS:
    g = r[A0_CTRL]["1"][p]
    xs += [PX[PS.index(p)]] * len(g["G_conf"]); ys += g["G_conf"]
for p in PS:
    g = r[A0_CTRL]["2"][p]
    xs += [PX[PS.index(p)]] * len(g["G_conf"]); ys += g["G_conf"]
ax.scatter(xs, ys, s=12, marker="x", color="0.45",
           label="$A_0$=0.1 (control)", zorder=2)
ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=1)
ax.set_xlabel("site-removal probability $p$")
ax.set_ylabel("conductance $G$  [$e^2/h$]")
ax.set_title("(a)  conductance, 960 devices")
ax.set_xlim(-0.03, 0.55); ax.set_ylim(0.4, 2.3)
ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.02, 0.37),
          framealpha=0.5, handletextpad=0.3)

# ---- (b) P_conf scatter ----
ax = axes[1]
for a0 in A0_WIN:
    for dep in DEPS:
        xs, ys = [], []
        for p in PS:
            g = r[a0][dep][p]
            xs += [PX[PS.index(p)]] * len(g["P_conf"]); ys += g["P_conf"]
        ax.scatter(xs, ys, s=14, marker=MARK[dep], facecolor="none",
                   edgecolor=COL[a0], linewidth=0.9, zorder=3)
xs, ys = [], []
for dep in DEPS:
    for p in PS:
        g = r[A0_CTRL][dep][p]
        xs += [PX[PS.index(p)]] * len(g["P_conf"]); ys += g["P_conf"]
ax.scatter(xs, ys, s=12, marker="x", color="0.45", zorder=2,
           label="$A_0$=0.1 (control)")
ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=1)
ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=1)
ax.set_xlabel("site-removal probability $p$")
ax.set_ylabel("polarization $P$")
ax.set_title("(b)  all 720 window devices: $P=+1.0000$")
ax.set_xlim(-0.03, 0.55); ax.set_ylim(-0.35, 1.35)
ax.legend(fontsize=7, loc="center left", framealpha=0.5)

# ---- (c) degradation ----
ax = axes[2]
for a0 in A0_WIN:
    for dep in DEPS:
        dg = [1.0 - r[a0][dep][p]["G_avg"] for p in PS]
        ax.plot(PX, [max(x, 1e-6) for x in dg], marker=MARK[dep], ms=5,
                color=COL[a0], lw=1.2, label=f"$A_0$={a0}, depth {dep}")
ax.axhline(0.005, color="crimson", lw=1.0, ls="--")
ax.text(0.02, 0.005 * 1.6, "0.5% (Anderson $W\\leq$16 meV budget)",
        fontsize=7, color="crimson")
ax.set_yscale("log")
ax.set_xlabel("site-removal probability $p$")
ax.set_ylabel("conductance loss $1-\\langle G\\rangle$")
ax.set_title("(c)  worst loss 0.02%, $T_\\downarrow=0$ exact")
ax.set_xlim(-0.03, 0.55)
ax.legend(fontsize=6.5, ncol=2, loc="upper right", framealpha=0.5,
          columnspacing=0.7, handletextpad=0.4)
ax.set_ylim(3e-6, 8e-2)

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print("saved:", OUT)

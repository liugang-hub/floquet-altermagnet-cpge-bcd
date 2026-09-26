# -*- coding: utf-8 -*-
"""Report-quality figure for hex BCD/NHE response-surface 补算 (Ningbo fund foundation).

Reads data/hex_bcd_surface_{v1,d8,t30}.json (all T=0/T=30, delta=3/8 variants)
Panels (English-only labels):
  (a) tilt scan @ (A0=0.06,hw=150), delta=3, T=0: Dx_tot / Dx_spin vs tilt T;
      T=0 row ~1e-8 (mirror protected), T>=10 opens D_x.
  (b) (A0 x hw) response surface of Dx_tot  (tilt=30, delta=3, T=0)
  (c) (A0 x hw) response surface of Dx_spin (tilt=30, delta=3, T=0)
  (d) temperature fingerprint at same delta=8 & hw=150: Dx_spin vs A0,
      T=0 (d8) vs T=30 meV (t30): spin channel suppressed ~2 decades at RT.
Usage: run with kwant2/plain python in paper_h_qgeom; out figs/hex_bcd_surface_report.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGD = os.path.join(HERE, "figs")
os.makedirs(FIGD, exist_ok=True)

def load(tag):
    return json.load(open(os.path.join(DATA, "hex_bcd_surface_%s.json" % tag), encoding="utf-8"))

v1, d8, t30 = load("v1"), load("d8"), load("t30")

# ---- (a) tilt scan (v1) ----
ts = v1["tilt_scan"]
xT = np.array([r["T"] for r in ts])
sA_tot = np.array([r["Dx_tot"] for r in ts])
sA_sp = np.array([r["Dx_spin"] for r in ts])
# y-symmetric log-ish scale not needed; linear with mirror-zero annotation
# ---- (b,c) heatmaps (v1 grid) ----
def grid_mat(d, key):
    A0s = sorted(set(r["A0"] for r in d["grid"]))
    hws = sorted(set(r["hw"] for r in d["grid"]))
    M = np.full((len(A0s), len(hws)), np.nan)
    for r in d["grid"]:
        i = A0s.index(r["A0"]); j = hws.index(r["hw"])
        M[i, j] = r[key]
    return np.array(A0s), np.array(hws), M

gA, gH, Mtot = grid_mat(v1, "Dx_tot")
_,  _, Msp   = grid_mat(v1, "Dx_spin")

# ---- (d) temperature fingerprint: Dx_spin vs A0 at hw=150, delta=8 ----
def slice_hw(d, key="Dx_spin", hw=150.0):
    rows = [r for r in d["grid"] if abs(r["hw"] - hw) < 1e-9]
    rows.sort(key=lambda r: r["A0"])
    return np.array([r["A0"] for r in rows]), np.array([r[key] for r in rows])

a0d, sp_d8  = slice_hw(d8, "Dx_spin")
a0t, sp_t30 = slice_hw(t30, "Dx_spin")

fig = plt.figure(figsize=(13, 9.6))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30, left=0.075, right=0.95,
                      top=0.92, bottom=0.09)

# (a)
ax = fig.add_subplot(gs[0, 0])
ax.plot(xT, sA_sp, "o-", color="tab:red", ms=5, lw=1.5, label="$D_x$ (spin)")
ax.plot(xT, sA_tot, "s--", color="tab:blue", ms=5, lw=1.5, label="$D_x$ (total)")
ax.axhline(0, color="k", lw=0.6)
ax.text(23, -31.5, "$T=0$: $D_x\\sim10^{-8}$\n(mirror protected)",
        fontsize=8.5, ha="left", va="top",
        bbox=dict(fc="white", ec="0.7", alpha=0.55, lw=0.7))
ax.set_ylim(-38, 8)
ax.set_xlabel("tilt $T$ (meV$\\cdot$nm)")
ax.set_ylabel("$D_x$")
ax.set_title("(a) tilt scan  ($A_0=0.06$, $\\hbar\\omega=150$ meV, $\\delta=3$, $T=0$)")
ax.legend(fontsize=9, loc="upper right", framealpha=0.5)
ax.grid(alpha=0.25)

# (b)
ax = fig.add_subplot(gs[0, 1])
im = ax.pcolormesh(gH, gA, Mtot, shading="auto", cmap="RdBu_r")
cb = fig.colorbar(im, ax=ax)
cb.set_label("$D_x$ (total)")
ax.set_xlabel("$\\hbar\\omega$ (meV)"); ax.set_ylabel("$A_0$ (nm$^{-1}$)")
ax.set_title("(b) response surface: total")
ax.grid(alpha=0.2, lw=0.4)

# (c)
ax = fig.add_subplot(gs[1, 0])
im = ax.pcolormesh(gH, gA, Msp, shading="auto", cmap="RdBu_r")
cb = fig.colorbar(im, ax=ax)
cb.set_label("$D_x$ (spin)")
ax.set_xlabel("$\\hbar\\omega$ (meV)"); ax.set_ylabel("$A_0$ (nm$^{-1}$)")
ax.set_title("(c) response surface: spin  ($A_0\\gtrsim0.08$ splits from total)")
ax.grid(alpha=0.2, lw=0.4)

# (d)
ax = fig.add_subplot(gs[1, 1])
ax.plot(a0d, sp_d8, "o-", color="tab:red", ms=5, lw=1.5, label="$k_BT=0$ (T=0)")
ax.plot(a0t, sp_t30, "s--", color="tab:green", ms=5, lw=1.5, label="$k_BT=30$ meV (RT)")
ax.axhline(0, color="k", lw=0.6)
ax.text(0.004, -5.9, "$A_0=0.06$: $\\approx -2.8 \\to -0.025$\n(2-decade suppression)",
        fontsize=8.5, ha="left", va="top",
        bbox=dict(fc="white", ec="0.7", alpha=0.55, lw=0.7))
ax.set_ylim(-7.3, 1.3)
ax.set_xlabel("$A_0$ (nm$^{-1}$)")
ax.set_ylabel("$D_x$ (spin)")
ax.set_title("(d) temperature fingerprint  ($\\hbar\\omega=150$ meV, $\\delta=8$, tilt=30)")
ax.legend(fontsize=9, loc="upper right", framealpha=0.5)
ax.grid(alpha=0.25)

fig.suptitle("hex altermagnet: BCD / nonlinear-Hall response surface  (Floquet, $J_0=160$ meV)",
             fontsize=12)
out = os.path.join(FIGD, "hex_bcd_surface_report.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("[fig] %s" % out)

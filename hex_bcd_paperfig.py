# -*- coding: utf-8 -*-
"""Paper-grade Fig. 5 (response surfaces) for Paper H PRB version.

Reads data/hex_bcd_surface_{v1,d8,t30}.json (all numbers cross-checked).
Panels:
  (a) tilt scan at delta=3 meV, T=0 (v1): Dx_tot & Dx_spin vs tilt;
      mirror-protected zero at tilt=0, spin-channel sign flip between 20 and 30.
  (b) room-temperature tilt linearity (t30, delta=8): Dx_tot linear,
      R2=0.99953; Dx_spin suppressed ~2 decades.
  (c) (A0,hw) response surface of Dx_spin (v1, delta=3, T=0), 7x7 grid;
      charge/spin channel separation for A0 >= 0.08.
  (d) temperature fingerprint at delta=8, hw=150, tilt=30: Dx vs A0 for
      kBT=0 (d8) and kBT=30 meV (t30); charge retains ~7%, spin ~1%.

Output: figs/fig5_response.png  (300 dpi) and copy to paper_h_main/figs/.
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIGD = os.path.join(BASE, "figs")
MAIN_FIGS = os.path.join(BASE, "paper_h_main", "figs")

def load(tag):
    return json.load(open(os.path.join(DATA, "hex_bcd_surface_%s.json" % tag),
                          encoding="utf-8"))

v1, d8, t30 = load("v1"), load("d8"), load("t30")

def series(scan, key_tot=True):
    T = np.array([r["T"] for r in scan])
    tot = np.array([r["Dx_up"] + r["Dx_dn"] for r in scan])
    spin = np.array([r["Dx_up"] - r["Dx_dn"] for r in scan])
    return T, tot, spin

def grid_map(g, field):
    """Return (A0s, hws, Z) with Z[i,j] = value at (A0_i, hw_j)."""
    a0s = sorted({r["A0"] for r in g})
    hws = sorted({r["hw"] for r in g})
    Z = np.full((len(a0s), len(hws)), np.nan)
    for r in g:
        i = a0s.index(r["A0"]); j = hws.index(r["hw"])
        if field == "tot":
            Z[i, j] = r["Dx_up"] + r["Dx_dn"]
        elif field == "spin":
            Z[i, j] = r["Dx_up"] - r["Dx_dn"]
        else:
            Z[i, j] = r[field]
    return np.array(a0s), np.array(hws), Z

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "font.family": "sans-serif",
})

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
(axa, axb), (axc, axd) = axes

# ---------------- (a) tilt scan, delta=3, T=0 ----------------
T, tot, spin = series(v1["tilt_scan"])
axa.plot(T, tot, "s-", color="tab:blue", ms=4.5, lw=1.4, label=r"$D_x^{\rm tot}$")
axa.plot(T, spin, "o--", color="tab:red", ms=4.5, lw=1.4, label=r"$D_x^{\rm spin}$")
axa.axhline(0, color="k", lw=0.6)
axa.annotate("mirror-protected zero\n($|D_x|<10^{-8}$)",
             xy=(0, 0), xytext=(48.5, -33.5), fontsize=7.5, ha="right",
             va="bottom",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="0.45",
                             connectionstyle="arc3,rad=0.18"))
axa.annotate("sign flip ($20$ to $30$)",
             xy=(36, 5.0), fontsize=7.5, color="tab:red", ha="center")
axa.set_xlabel(r"tilt $T_{\rm tilt}$ (meV$\,$nm)")
axa.set_ylabel(r"$D_x$ (model units)")
axa.set_title(r"(a) tilt scan ($\delta=3$ meV, $k_BT=0$)", fontsize=9.5)
axa.legend(loc="center right", bbox_to_anchor=(0.98, 0.62), frameon=False)
axa.grid(alpha=0.25, lw=0.4)

# ---------------- (b) room-temperature tilt linearity ----------------
T2, tot2, spin2 = series(t30["tilt_scan"])
axb.plot(T2, tot2, "s-", color="tab:green", ms=4.5, lw=1.4,
         label=r"$D_x^{\rm tot}$ ($k_BT=30$ meV)")
axb.plot(T2, spin2, "^:", color="tab:purple", ms=4.5, lw=1.4,
         label=r"$D_x^{\rm spin}$ ($k_BT=30$ meV)")
sl, ic = t30["tilt_fit"]["Dx_tot"]["slope"], t30["tilt_fit"]["Dx_tot"]["intercept"]
r2 = t30["tilt_fit"]["Dx_tot"]["R2_linear"]
xx = np.linspace(0, 50, 10)
axb.plot(xx, ic + sl * xx, "k--", lw=1.0,
         label=r"linear fit, $R^2=%.4f$" % r2)
axb.set_xlabel(r"tilt $T_{\rm tilt}$ (meV$\,$nm)")
axb.set_ylabel(r"$D_x$ (model units)")
axb.set_title(r"(b) room-temperature tilt response ($\delta=8$ meV)", fontsize=9.5)
axb.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
axb.grid(alpha=0.25, lw=0.4)

# ---------------- (c) (A0,hw) surface of Dx_spin ----------------
a0s, hws, Zs = grid_map(v1["grid"], "spin")
A0m, HWm = np.meshgrid(a0s, hws, indexing="ij")
vmax = 16.0
pc = axc.pcolormesh(HWm, A0m, Zs, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    shading="nearest")
cb = fig.colorbar(pc, ax=axc, pad=0.02)
cb.set_label(r"$D_x^{\rm spin}$", fontsize=8.5)
cb.ax.tick_params(labelsize=7.5)
axc.axhline(0.08, color="k", ls="--", lw=0.8)
axc.text(252, 0.078, "channels separated by $A_0=0.08$", fontsize=7,
         ha="right", va="top")
axc.set_xlabel(r"$\hbar\omega$ (meV)")
axc.set_ylabel(r"$A_0$ (nm$^{-1}$)")
axc.set_title(r"(c) spin BCD surface ($\delta=3$ meV, $k_BT=0$)", fontsize=9.5)

# ---------------- (d) temperature fingerprint ----------------
def grid_row(g, hw):
    a0s = sorted({r["A0"] for r in g if r["hw"] == hw})
    tot = [r["Dx_up"] + r["Dx_dn"] for r in sorted(g, key=lambda r: r["A0"]) if r["hw"] == hw]
    spin = [r["Dx_up"] - r["Dx_dn"] for r in sorted(g, key=lambda r: r["A0"]) if r["hw"] == hw]
    return np.array(a0s), np.array(tot), np.array(spin)

a0d, totd, spind = grid_row(d8["grid"], 150.0)
a0t, tott, spint = grid_row(t30["grid"], 150.0)
axd.plot(a0d, totd, "s-", color="tab:red", ms=4, lw=1.3,
         label=r"$D_x^{\rm tot}$, $k_BT=0$")
axd.plot(a0d, spind, "o--", color="tab:red", ms=4, lw=1.2, alpha=0.75,
         label=r"$D_x^{\rm spin}$, $k_BT=0$")
axd.plot(a0t, tott, "s-", color="tab:green", ms=4, lw=1.3,
         label=r"$D_x^{\rm tot}$, $k_BT=30$ meV")
axd.plot(a0t, spint, "^:", color="tab:green", ms=4, lw=1.2, alpha=0.85,
         label=r"$D_x^{\rm spin}$, $k_BT=30$ meV")
axd.axhline(0, color="k", lw=0.6)
axd.set_xlabel(r"$A_0$ (nm$^{-1}$)")
axd.set_ylabel(r"$D_x$ (model units)")
axd.set_title(r"(d) temperature fingerprint ($\delta=8$ meV, $\hbar\omega=150$)",
              fontsize=9.5)
axd.legend(loc="center left", frameon=True, framealpha=0.9,
           edgecolor="none", ncol=1)
axd.grid(alpha=0.25, lw=0.4)

fig.tight_layout(pad=0.6)
out = os.path.join(FIGD, "fig5_response.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("[fig] %s" % out)
try:
    shutil.copy(out, os.path.join(MAIN_FIGS, "fig5_response.png"))
    print("[copy] -> paper_h_main/figs/fig5_response.png")
except Exception as e:
    print("[copy skipped]", e)

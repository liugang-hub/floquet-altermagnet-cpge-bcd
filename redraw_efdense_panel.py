"""Redraw the dense EF-scan Fig.3 from official json (no recomputation).

Usage: python redraw_efdense_panel.py
Reads data/hex_ef_dense.json and renders
figs/hex_ef_dense_panel.png with Times New Roman fonts and (a)-(c) labels.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_pub_fonts():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def add_panel_label(ax, label, x=0.03, y=0.90):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            fontname="Times New Roman", color="black",
            zorder=10)


set_pub_fonts()

d = json.load(open("data/hex_ef_dense.json"))
params = d["params"]
curves = d["curves"]
A0 = params["A0"]
J0 = params["J0"]
TILT = params["TILT"]
NK = params["NK"]
window = params["window_meV"]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

for Tk in [0.0, 30.0]:
    rows = curves[f"T{Tk:g}"]
    ds = [r["delta"] for r in rows]
    lbl = rf"$T={Tk:g}$ meV"
    axes[0].plot(ds, [r["Dx_spin"] for r in rows], "o-", lw=1.2, ms=2,
                 label=lbl)
    axes[1].plot(ds, [r["Dx_up"] for r in rows], "o-", color="#c0392b",
                 lw=1.5, ms=3, label=lbl + r", $\uparrow$")
    axes[1].plot(ds, [r["Dx_dn"] for r in rows], "s-", color="#1f4e79",
                 lw=1.5, ms=3, label=lbl + r", $\downarrow$")
    axes[2].plot(ds, [r["occ_up"] for r in rows], "o-", color="#c0392b",
                 lw=1.3, ms=2.5, label=lbl + r", occ$_\uparrow$")
    axes[2].plot(ds, [r["occ_dn"] for r in rows], "s-", color="#1f4e79",
                 lw=1.3, ms=2.5, label=lbl + r", occ$_\downarrow$")

axes[0].axhline(0, color="gray", ls=":", lw=0.8)
axes[0].axvspan(0, window, color="gold", alpha=0.18,
                label="single-spin window")
axes[0].set_xlabel(r"$\delta=E_{\rm top}-E_F$ (meV)")
axes[0].set_ylabel(r"$D_x^{\rm spin}$")
axes[0].set_title(rf"$D_x^s(\delta)$, $A_0={A0:.2f}$")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)
add_panel_label(axes[0], "a")

axes[1].axhline(0, color="gray", ls=":", lw=0.8)
axes[1].axvspan(0, window, color="gold", alpha=0.18)
axes[1].set_xlabel(r"$\delta$ (meV)")
axes[1].set_ylabel(r"$D_x$")
axes[1].set_title("Spin-resolved BCD")
axes[1].legend(fontsize=6)
axes[1].grid(alpha=0.3)
add_panel_label(axes[1], "b", y=0.84)

axes[2].axhline(1.0, color="gray", ls=":", lw=0.8)
axes[2].axvspan(0, window, color="gold", alpha=0.18)
axes[2].set_xlabel(r"$\delta$ (meV)")
axes[2].set_ylabel("occupancy")
axes[2].set_title("Fermi-surface character")
axes[2].legend(fontsize=6)
axes[2].grid(alpha=0.3)
add_panel_label(axes[2], "c")

fig.suptitle(rf"Paper H -- dense EF scan of spin BCD, $J_0={J0:.0f}$ meV, "
             rf"tilt={TILT:.0f}$\,$meV, window={window:.2f}$\,$meV",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("figs/hex_ef_dense_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("Redrawn figs/hex_ef_dense_panel.png from data/hex_ef_dense.json")

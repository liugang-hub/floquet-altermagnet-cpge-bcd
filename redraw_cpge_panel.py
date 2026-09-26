"""Redraw the CPGE panel figure from the OFFICIAL json (no computation).

Usage: python redraw_cpge_panel.py
Reads data/hex_cpge.json (currently the NK=200 official run) and renders
figs/hex_cpge_panel.png with publication fonts and (a)-(f) labels.

Shared legend: all six panels plot the same three curves (up/down/total),
so a single figure-level legend is placed at the top of the figure.
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
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def add_panel_label(ax, label, x=0.03, y=0.92):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            fontname="Times New Roman", color="black",
            zorder=10)


set_pub_fonts()

d = json.load(open("data/hex_cpge.json"))
res = d["results"]
params = d["params"]
J0 = params.get("J0", 160.0)
TILT = params.get("TILT", 30.0)

Ms = list(res.keys())
A0s = list(res[Ms[0]].keys())
nM, nA = len(Ms), len(A0s)

# extra top headroom for the figure-level shared legend + suptitle
fig, axes = plt.subplots(nM, nA, figsize=(3.2 * nA + 0.6, 2.6 * nM + 1.0),
                         sharex=True, sharey=False)
if nM == 1:
    axes = axes.reshape(1, -1)

panel_labels = ["a", "b", "c", "d", "e", "f"]
label_idx = 0
for iM, Mk in enumerate(Ms):
    for iA, A0k in enumerate(A0s):
        ax = axes[iM, iA]
        dat = res[Mk][A0k]
        om = np.array(dat["omega"])
        ax.plot(om, np.array(dat["beta_up"]) * 1e3, "r-", lw=1.2, label=r"$\uparrow$")
        ax.plot(om, np.array(dat["beta_dn"]) * 1e3, "b-", lw=1.2, label=r"$\downarrow$")
        ax.plot(om, np.array(dat["beta_tot"]) * 1e3, "k--", lw=1.5, label="total")
        ax2 = ax.twinx()
        ax2.plot(om, np.array(dat["P_cpge"]), "g:", lw=1.0, alpha=0.8)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_ylabel(r"$P_{\rm cpge}$", fontsize=8, color="green")
        ax.axhline(0, color="gray", ls=":", lw=0.6)
        ax.set_xlim(0, 60)
        if iM == nM - 1:
            ax.set_xlabel(r"$\hbar\omega$ (meV)")
        if iA == 0:
            ax.set_ylabel(r"$\beta^{xy}\,(\times10^{-3})$")
        if iM == 0:
            ax.set_title(f"$A_0={float(A0k[3:]):.3f}$ nm$^{{-1}}$")
        add_panel_label(ax, panel_labels[label_idx])
        label_idx += 1

# figure-level shared legend at the very top (all six panels share the same
# three curve labels); first three lines of panel (a) are up/dn/tot curves.
fig.legend(handles=axes[0, 0].lines[:3],
           loc="lower center", bbox_to_anchor=(0.5, 0.915),
           ncol=3, frameon=False, fontsize=9,
           handlelength=2.2, columnspacing=1.6)

fig.suptitle(f"Paper H -- hexagonal-lattice CPGE, $J_0={J0:.0f}$ meV, "
             f"tilt={TILT:.0f} meV$\\cdot$nm (NK={params.get('NK', 200)}, official)",
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.885])
plt.savefig("figs/hex_cpge_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("Redrawn figs/hex_cpge_panel.png from official json "
      f"(M={Ms}, A0s={A0s}, NK={params.get('NK')})")

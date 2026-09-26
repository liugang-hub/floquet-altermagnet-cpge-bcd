"""Redraw the full BCD/NHE Fig.3 from the official NK=400 json (no recomputation).

Usage: python redraw_bcd_panel.py
Reads data/hex_bcd_nhe.json (official run) and renders
figs/hex_bcd_nhe_panel.png with publication fonts and (a)-(d) labels.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hex_model as hm


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


def add_panel_label(ax, label, x=0.03, y=0.90):
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
            fontname="Times New Roman", color="black",
            zorder=10)


set_pub_fonts()

d = json.load(open("data/hex_bcd_nhe.json"))
rows = d["ef_scan"]
a0s = d["a0_scan"]
params = d["params"]

A0_EF = params["A0_ef"]
J0 = params["J0"]
M0 = params["M0"]
A = params["A"]
TILT = 30.0

fig = plt.figure(figsize=(12.5, 3.6))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

# ---- panel (a): Dx_tot vs delta for control rows ---------------------------
CTRL = [(0.0, 0.0), (0.0, 30.0), (30.0, 30.0)]
for (Ttmp, Ttilt) in CTRL:
    sub = [r for r in rows if r["T"] == Ttmp and r["TILT"] == Ttilt]
    lbl = rf"$T={Ttmp:.0f}$, tilt={Ttilt:.0f}"
    ax1.plot([r["delta"] for r in sub], [r["Dx_tot"] for r in sub],
             "o-", lw=1.3, ms=3.5, label=lbl)
ax1.axhline(0, color="gray", ls=":", lw=0.8)
ax1.set_xlabel(r"$\delta=E_{\rm top}-E_F$ (meV)")
ax1.set_ylabel(r"$D_x^{\rm tot}$")
ax1.set_title(rf"BCD vs depth, $A_0={A0_EF:.2f}$")
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3, ls="--")
add_panel_label(ax1, "a", y=0.84)

# ---- panel (b): spin-resolved Dx at T=30, tilt=30 --------------------------
sub = [r for r in rows if r["T"] == 30.0 and r["TILT"] == 30.0]
ax2.plot([r["delta"] for r in sub], [r["Dx_up"] for r in sub], "ro-",
         lw=1.3, ms=3.5, label=r"$D_x^{\uparrow}$")
ax2.plot([r["delta"] for r in sub], [r["Dx_dn"] for r in sub], "bo-",
         lw=1.3, ms=3.5, label=r"$D_x^{\downarrow}$")
ax2.plot([r["delta"] for r in sub], [r["Dx_spin"] for r in sub], "g^-",
         lw=1.3, ms=3.5, label=r"$D_x^{\rm spin}$")
ax2.axhline(0, color="gray", ls=":", lw=0.8)
ax2.set_xlabel(r"$\delta$ (meV)")
ax2.set_ylabel(r"$D_x$")
ax2.set_title(r"Spin-resolved BCD, $T$=30 meV")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3, ls="--")
add_panel_label(ax2, "b")

# ---- panel (c): A0 scan at metallic EF ------------------------------------
EF_metal = a0s[0]["EF"]
ax3.plot([r["A0"] for r in a0s], [r["Dx_up"] for r in a0s], "ro-",
         lw=1.3, ms=3.5, label=r"$D_x^{\uparrow}$")
ax3.plot([r["A0"] for r in a0s], [r["Dx_dn"] for r in a0s], "bo-",
         lw=1.3, ms=3.5, label=r"$D_x^{\downarrow}$")
ax3.plot([r["A0"] for r in a0s], [r["Dx_tot"] for r in a0s], "k--s",
         lw=1.2, ms=3, label=r"$D_x^{\rm tot}$")
ax3.plot([r["A0"] for r in a0s], [r["Dx_spin"] for r in a0s], "g^-",
         lw=1.3, ms=3.5, label=r"$D_x^{\rm spin}$")
ax3.axhline(0, color="gray", ls=":", lw=0.8)
ax3.axvline(0.12, color="gray", ls=":", lw=0.8)
ax3.set_xlabel(r"$A_0$ (nm$^{-1}$)")
ax3.set_ylabel(r"$D_x$")
ax3.set_title(rf"Floquet tuning, $E_F={EF_metal:+.1f}$ meV")
ax3.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 0.06))
ax3.grid(alpha=0.3, ls="--")
add_panel_label(ax3, "c")

# ---- panel (d): Berry-curvature spin character at the A0-scan point -------
At_w, Mt_w = hm.floquet_params(hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, M0,
                               hm.HW_PARAM, A0_EF)
NK = 400
kx, ky, mask, _ = hm.hex_mesh(NK)


def occ_step(Ev, EF, width=0.02):
    return 0.5 * (1.0 - np.tanh((Ev - EF) / width))


occ_up = occ_step(hm.valence_energy(kx, ky, "up", At_w, Mt_w, A0_EF, TILT, J0),
                  EF_metal)
occ_dn = occ_step(hm.valence_energy(kx, ky, "dn", At_w, Mt_w, A0_EF, TILT, J0),
                  EF_metal)

th = 0.5
up_only = mask & (occ_up > th) & (occ_dn < th)
dn_only = mask & (occ_up < th) & (occ_dn > th)
both = mask & (occ_up > th) & (occ_dn > th)

R = hm.R_HEX
hex_ang = np.linspace(0, 2 * np.pi, 7)
hx, hy = R * np.cos(hex_ang + np.pi / 6.0), R * np.sin(hex_ang + np.pi / 6.0)
ax4.plot(hx, hy, "k-", lw=1.0)

step = max(1, NK // 200)
if up_only.any():
    ax4.scatter(kx[up_only][::step], ky[up_only][::step],
                c="red", s=3.0, alpha=0.7, label=r"$\uparrow$ only")
if dn_only.any():
    ax4.scatter(kx[dn_only][::step], ky[dn_only][::step],
                c="blue", s=3.0, alpha=0.7, label=r"$\downarrow$ only")
if both.any():
    ax4.scatter(kx[both][::step], ky[both][::step],
                c="purple", s=1.0, alpha=0.4, label="both")

ax4.set_aspect("equal")
ax4.set_xlabel(r"$k_x$ (nm$^{-1}$)")
ax4.set_ylabel(r"$k_y$ (nm$^{-1}$)")
ax4.set_title(rf"Berry spin character, $E_F={EF_metal:+.1f}$ meV")
ax4.legend(fontsize=5, loc="upper right", frameon=True, framealpha=0.85,
           edgecolor="none", borderpad=0.25, labelspacing=0.15,
           handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)
add_panel_label(ax4, "d")

fig.suptitle(f"hexagonal-lattice BCD/NHE: $J_0={J0:.0f}$ meV, "
             f"$M={M0:+.0f}$ meV, $A={A:.0f}$ meV", fontsize=12, y=0.965)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figs/hex_bcd_nhe_panel.png", dpi=300, bbox_inches="tight")
plt.close()

sub30 = [r for r in rows if r["T"] == 30.0 and r["TILT"] == 30.0]
print("Redrawn figs/hex_bcd_nhe_panel.png from official json")
print(f"  (panel b: T=30 Dx_spin range "
      f"[{min(r['Dx_spin'] for r in sub30):+.3f}, "
      f"{max(r['Dx_spin'] for r in sub30):+.3f}])")

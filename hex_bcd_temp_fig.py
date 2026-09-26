# -*- coding: utf-8 -*-
"""Standalone replot of figs/hex_bcd_temp_temp.png from archived json
(no recompute). Legend frames translucent (report figure standard)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "data", "hex_bcd_temp_temp.json")
OUT = os.path.join(HERE, "figs", "hex_bcd_temp_temp.png")

d = json.load(open(J, encoding="utf-8"))
rows = d["rows"]
p = d["params"]
GAMMAS = sorted(set(r["gamma"] for r in rows))
colors = {0.0: "k", 1.0: "tab:blue", 2.0: "tab:red", 4.0: "tab:green"}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
for ax, key, lab, tag in ((axes[0], "Dx_tot", r"$D_x^{\rm tot}$ (charge NHE)", "a"),
                          (axes[1], "Dx_spin", r"$D_x^{\rm spin}$ (spin NHE)", "b")):
    for gam in GAMMAS:
        sub = [r for r in rows if r["gamma"] == gam]
        ax.plot([r["T"] for r in sub], [r[key] for r in sub], "o-",
                lw=1.4, ms=4, color=colors.get(gam, None),
                label=r"$\gamma=%.0f$ meV" % gam)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$k_BT$ (meV)")
    ax.set_ylabel(lab)
    ax.set_title(r"(%s) $A_0=0.06$, $\hbar\omega=150$, tilt$=30$, "
                 r"$\delta=8$ meV" % tag)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.grid(alpha=0.3)
fig.suptitle("BCD vs temperature and scattering broadening "
             r"($J_0=%.0f$ meV, $N_K=%d$)" % (p["J0"], p["NK"]), y=0.96)
plt.tight_layout(rect=[0, 0, 1, 0.925])
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("[fig] %s" % OUT)

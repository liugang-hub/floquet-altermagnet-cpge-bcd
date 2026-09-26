# -*- coding: utf-8 -*-
"""Paper-grade Fig. 6 (ribbon device: disorder x dephasing) for Paper H PRB.

Reads hex_ribbon_dev/data/hex_rib_disorder_dp_disdp.json (official, NC=20).
Same panels as hex_rib_dp_analyze.py but without suptitle, PRB-friendly fonts.
Output: figs/fig6_device.png (300 dpi) + copy to paper_h_main/figs/.
"""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
J = os.path.join(HERE, "data", "hex_rib_disorder_dp_disdp.json")
MAIN_FIGS = os.path.join(BASE, "paper_h_main", "figs")

d = json.load(open(J, encoding="utf-8"))
res = d["results"]
A0s = sorted(res, key=float)

def get(A0, W, g, key="G_avg"):
    for kk in res[A0][W]:
        if abs(float(kk) - float(g)) < 1e-9:
            return float(res[A0][W][kk][key])
    raise KeyError((A0, W, g))

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "font.family": "sans-serif",
})

fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.42,
                      left=0.08, right=0.985, top=0.95, bottom=0.12)
wstyle = {"0": ("-", "o", "tab:blue"), "8": ("--", "s", "tab:red"),
          "16": (":", "^", "tab:green")}

def panel_g(ax, A0, tag):
    for W in sorted(res[A0], key=float):
        xs = [float(g) for g in sorted(res[A0][W], key=float)]
        ys = [get(A0, W, g) for g in xs]
        err = [get(A0, W, g, "G_std") for g in xs]
        ls, mk, c = wstyle.get(W, ("-", "o", "k"))
        ax.errorbar(xs, ys, yerr=err, fmt=mk + ls, color=c, ms=3.6, lw=1.2,
                    capsize=2, label="$W=%s$ meV" % W)
    ax.set_yscale("log")
    ax.set_xlabel(r"dephasing $\gamma$ (meV)")
    ax.set_ylabel(r"$G$ ($e^2/h$)")
    ax.set_title("(%s) $A_0=%s$ nm$^{-1}$" % (tag, A0), fontsize=9.5)
    ax.set_ylim(5e-3, 3.0)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    if float(A0) >= 0.115:
        ax.text(0.96, 0.95, r"$P=+1.0000$ all $\gamma$" "\n" r"($T_{\downarrow}\equiv 0$)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(fc="white", ec="0.4", alpha=0.6, lw=0.5))

axa = fig.add_subplot(gs[0, 0]); panel_g(axa, "0.115", "a")
axb = fig.add_subplot(gs[0, 1]); panel_g(axb, "0.118", "b")
axc = fig.add_subplot(gs[0, 2]); panel_g(axc, "0.12", "c")
axd = fig.add_subplot(gs[1, 0]); panel_g(axd, "0.1", "d")
axd.text(0.05, 0.05, "QSH control\n($P=0$)", transform=axd.transAxes,
         ha="left", va="bottom", fontsize=7,
         bbox=dict(fc="white", ec="0.4", alpha=0.6, lw=0.8))

axe = fig.add_subplot(gs[1, 1])
for A0 in A0s:
    xs = [float(g) for g in sorted(res[A0]["0"], key=float)]
    ps = [get(A0, "0", g, "P_avg") for g in xs]
    c = "tab:red" if float(A0) >= 0.115 else "tab:gray"
    axe.plot(xs, ps, "o-", color=c, ms=3.6, lw=1.2, label="$A_0=%s$" % A0)
axe.axhline(1.0, color="tab:green", ls="--", lw=0.9)
axe.set_xlabel(r"dephasing $\gamma$ (meV)")
axe.set_ylabel(r"$P$")
axe.set_title(r"(e) $P$ vs $\gamma$ ($W=0$)", fontsize=9.5)
axe.set_ylim(-0.35, 1.12)
axe.grid(alpha=0.25)
axe.legend(fontsize=5.5, ncol=2, loc="center", framealpha=0.5,
           columnspacing=0.7, handletextpad=0.3, handlelength=1.1,
           labelspacing=0.35, borderpad=0.3)

axf = fig.add_subplot(gs[1, 2])
for A0 in A0s:
    for W in res[A0]:
        G0 = get(A0, W, 0)
        xs = [float(g) for g in sorted(res[A0][W], key=float)]
        ys = [max(get(A0, W, g) / G0, 1e-3) for g in xs]
        axf.plot(xs, ys, "-", color="tab:gray", lw=0.5, alpha=0.6)
g0ref = get("0.118", "0", 0)
xs = [float(g) for g in sorted(res["0.118"]["0"], key=float)]
ym = np.array([get("0.118", "0", g) / g0ref for g in xs])
axf.plot(xs, ym, "o-", color="tab:blue", ms=4, lw=1.5,
         label="$A_0=0.118$, $W=0$")
al = []
for A0 in A0s:
    for W in res[A0]:
        G0 = get(A0, W, 0)
        if G0 <= 1e-6:
            continue
        gx = [float(g) for g in sorted(res[A0][W], key=float) if float(g) > 0]
        gy = [np.log(max(get(A0, W, g), 1e-12) / G0) for g in gx]
        if len(gx) >= 2:
            al.append(-np.polyfit(gx, gy, 1)[0])
am = float(np.mean(al))
xg = np.linspace(min(xs), max(xs), 50)
axf.plot(xg, np.exp(-am * xg), "k--", lw=1.2,
         label=r"$e^{-\alpha\gamma}$, $\alpha=%.2f$ meV$^{-1}$" % am)
axf.set_yscale("log")
axf.set_xlabel(r"dephasing $\gamma$ (meV)")
axf.set_ylabel(r"$G(\gamma)/G(0)$")
axf.set_title(r"(f) normalized collapse (all $A_0$, $W$)", fontsize=9.5)
axf.set_ylim(5e-3, 1.5)
axf.grid(alpha=0.25, which="both", lw=0.4)
axf.legend(fontsize=6, loc="upper right", framealpha=0.5)

out = os.path.join(HERE, "figs", "fig6_device.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("[fig] %s" % out)
try:
    shutil.copy(out, os.path.join(MAIN_FIGS, "fig6_device.png"))
    print("[copy] -> paper_h_main/figs/fig6_device.png")
except Exception as e:
    print("[copy skipped]", e)

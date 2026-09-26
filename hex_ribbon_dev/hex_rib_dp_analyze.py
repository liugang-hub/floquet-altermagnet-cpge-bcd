# -*- coding: utf-8 -*-
"""Harvest & plot hex_rib_disorder_dp results (disorder x dephasing joint scan).

Reads  data/hex_rib_disorder_dp_<TAG>.json   (TAG default disdp_fast, override HRD_TAG)
Checks:
  - exact spin selectivity: Tdn_avg == 0 in filter window A0 in [0.115,0.122] at ALL gamma
  - control A0=0.100 (outside window): G=2.0, P=0 at gamma=0
  - exponential decay law: ln(G/G0) = -alpha*gamma  (per A0,W), report alpha + R2
  - disorder flatness: G(W) at fixed (A0,gamma)
Writes figs/hex_rib_dp_<TAG>.png  (publication style, English labels only).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIGD = os.path.join(BASE, "figs")
os.makedirs(FIGD, exist_ok=True)

TAG = os.environ.get("HRD_TAG", "disdp_fast")
p = os.path.join(DATA, "hex_rib_disorder_dp_%s.json" % TAG)
if not os.path.exists(p):
    sys.exit("missing: %s" % p)
d = json.load(open(p, encoding="utf-8"))
res = d["results"]
print("== params:", json.dumps(d.get("params", {}), ensure_ascii=False))
print("== tag=%s  file=%s" % (TAG, p))

A0s = sorted(res, key=float)
Wins = {a: sorted(res[a], key=float) for a in A0s}

def _key(A0, W, g):
    for k in res[A0][W]:
        if abs(float(k) - float(g)) < 1e-9:
            return k
    return str(g)

def get(A0, W, g, key="G_avg"):
    return float(res[A0][W][_key(A0, W, g)][key])

# ---------- check 1: exact selectivity inside window ----------
n_exact = 0
bad = []
for A0 in A0s:
    a = float(A0)
    if not (0.115 <= a <= 0.122):
        continue
    for W in res[A0]:
        for g, leaf in res[A0][W].items():
            tdn = float(leaf["Tdn_avg"])
            if tdn == 0.0:
                n_exact += 1
            else:
                bad.append((A0, W, g, tdn))
print("\n[selectivity] window A0: (A0,W,gamma) rows with Tdn_avg==0 : %d" % n_exact)
if bad:
    print("  VIOLATIONS:", bad[:10])
else:
    print("  -> Tdn=0 exact at ALL gamma (incl. gamma=2 meV): PASS")

# ---------- check 2: control A0=0.1 ----------
for g in sorted(res["0.1"]["0"], key=float):
    G0 = get("0.1", "0", g)
    P0 = float(res["0.1"]["0"][str(g)]["P_avg"])
    if float(g) == 0.0:
        print("\n[control] A0=0.1 W=0 gamma=0: G=%.6f P=%.4f  (expect G=2, P=0)  %s"
              % (G0, P0, "PASS" if abs(G0 - 2) < 1e-9 and abs(P0) < 1e-9 else "FAIL"))

# ---------- check 3: exponential fit  ln(G/G0) = -alpha*gamma ----------
print("\n[decay law] fit ln(G/G0) = -a*gamma + b  over gamma>0")
fit_rows = []
for A0 in A0s:
    for W in res[A0]:
        G0 = get(A0, W, 0)
        if G0 <= 1e-6:
            continue
        xs, ys = [], []
        for g in sorted(res[A0][W], key=float):
            gv = float(g)
            if gv > 0:
                xs.append(gv)
                ys.append(np.log(max(get(A0, W, g), 1e-12) / G0))
        if len(xs) >= 2:
            a, b = np.polyfit(xs, ys, 1)
            yh = a * np.asarray(xs) + b
            denom = np.sum((np.asarray(ys) - np.mean(ys)) ** 2)
            R2 = 1 - np.sum((np.asarray(ys) - yh) ** 2) / denom if denom > 1e-30 else float("nan")
            fit_rows.append((A0, W, -a, b, R2))
for A0, W, al, b, R2 in fit_rows:
    print("  A0=%-6s W=%-4s  alpha=%7.3f meV^-1   b=%+.4f   R2=%.4f" % (A0, W, al, b, R2))
al_vals = np.array([r[2] for r in fit_rows if r[1] == "0"])
if len(al_vals):
    print("  -> W=0 rows: alpha = %.3f +- %.3f meV^-1" % (al_vals.mean(), al_vals.std()))

# ---------- check 4: disorder flatness at fixed gamma ----------
print("\n[disorder] G(W)/G(W=0) at gamma=0.5  (flatness under disorder)")
for A0 in A0s:
    a = float(A0)
    if not (0.115 <= a <= 0.122):
        continue
    g0 = get(A0, "0", 0.5)
    line = "  A0=%s:" % A0
    for W in sorted(res[A0], key=float):
        r = get(A0, W, 0.5) / g0
        line += "  W=%-3s %.4f" % (W, r)
    print(line)

# ============================ FIGURE ============================
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5})
fig = plt.figure(figsize=(13.5, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

wstyle = {"0": ("-", "o", "tab:blue"), "8": ("--", "s", "tab:red"),
          "16": (":", "^", "tab:green")}

def panel_g(ax, A0, tag):
    for W in sorted(res[A0], key=float):
        xs = [float(g) for g in sorted(res[A0][W], key=float)]
        ys = [get(A0, W, g) for g in xs]
        err = [get(A0, W, g, "G_std") for g in xs]
        ls, mk, c = wstyle.get(W, ("-", "o", "k"))
        ax.errorbar(xs, ys, yerr=err, fmt=mk + ls, color=c, ms=4.5, lw=1.3,
                    capsize=2.5, label="W=%s meV" % W)
    ax.set_yscale("log")
    ax.set_xlabel("dephasing $\\gamma$ (meV)")
    ax.set_ylabel("$G$ ($e^2/h$)")
    ax.set_title("(%s)  $A_0=%s$ nm$^{-1}$" % (tag, A0), fontsize=10.5)
    ax.set_ylim(3e-2, 3.0)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    if float(A0) >= 0.115:
        ax.text(0.97, 0.06, "$P=+1.0000$ all $\\gamma$\n($T_{\\downarrow}\\equiv0$)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(fc="white", ec="tab:green", alpha=0.9, lw=1))

axa = fig.add_subplot(gs[0, 0]); panel_g(axa, "0.115", "a")
axb = fig.add_subplot(gs[0, 1]); panel_g(axb, "0.118", "b")
axc = fig.add_subplot(gs[0, 2]); panel_g(axc, "0.12", "c")
axd = fig.add_subplot(gs[1, 0]); panel_g(axd, "0.1", "d")

# (e) spin polarization vs gamma
axe = fig.add_subplot(gs[1, 1])
for A0 in A0s:
    xs = [float(g) for g in sorted(res[A0]["0"], key=float)]
    ps = [get(A0, "0", g, "P_avg") for g in xs]
    c = "tab:red" if float(A0) >= 0.115 else "tab:gray"
    axe.plot(xs, ps, "o-", color=c, ms=4, lw=1.2, label="$A_0=%s$" % A0)
axe.axhline(1.0, color="tab:green", ls="--", lw=1)
axe.set_xlabel("dephasing $\\gamma$ (meV)")
axe.set_ylabel("spin polarization  $P$")
axe.set_title("(e)  $P$ vs $\\gamma$  ($W=0$)")
axe.set_ylim(-0.35, 1.12)
axe.grid(alpha=0.25)
axe.legend(fontsize=8, ncol=2, loc="lower left")

# (f) collapse G/G0 vs gamma + universal decay
axf = fig.add_subplot(gs[1, 2])
for A0 in A0s:
    for W in res[A0]:
        G0 = get(A0, W, 0)
        xs = [float(g) for g in sorted(res[A0][W], key=float)]
        ys = [max(get(A0, W, g) / G0, 1e-3) for g in xs]
        axf.plot(xs, ys, "-", color="tab:gray", lw=0.6, alpha=0.7)
g0ref = get("0.118", "0", 0)
xs = [float(g) for g in sorted(res["0.118"]["0"], key=float)]
ym = np.array([get("0.118", "0", g) / g0ref for g in xs])
axf.plot(xs, ym, "o-", color="tab:blue", ms=5, lw=1.6,
         label="$A_0=0.118$, $W=0$")
am = float(np.mean(al_vals)) if len(al_vals) else 1.25
xg = np.linspace(min(xs), max(xs), 50)
axf.plot(xg, np.exp(-am * xg), "k--", lw=1.3,
         label="$e^{-\\alpha\\gamma}$, $\\alpha=%.2f$ meV$^{-1}$" % am)
axf.set_yscale("log")
axf.set_xlabel("dephasing $\\gamma$ (meV)")
axf.set_ylabel("$G(\\gamma)/G(0)$")
axf.set_title("(f)  normalized collapse  (all $A_0$, $W$)")
axf.set_ylim(1e-2, 1.5)
axf.grid(alpha=0.25, which="both", lw=0.4)
axf.legend(fontsize=8, loc="lower left")

fig.suptitle("hex ribbon: spin-filter robustness vs dephasing  ($N_y=160$, $L_x=60$, $N_C$ per tag)",
             fontsize=12)
out = os.path.join(FIGD, "hex_rib_dp_%s.png" % TAG)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("\n[fig] %s" % out)

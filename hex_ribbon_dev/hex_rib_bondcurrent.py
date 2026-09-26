# -*- coding: utf-8 -*-
r"""
hex_rib_bondcurrent.py -- Paper H 补图: ribbon 自旋过滤器的实空间键电流图
================================================================================
目的 (2026-09-20, 主稿 Device 节机制可视化):
  用 kwant 散射态 + 逐键电流 J_{b->a} = 2 Im[psi_a^dag H_ab psi_b],
  把 "P=+1.000 单自旋过滤" 画成实空间电流分布:
    列 1: A0=0.100 nm^-1 (自旋简并 QAH 对照, C_up=C_dn=-1)
          -- up/dn 同边同向并行 (G=2, P=0)
    列 2: A0=0.118 nm^-1 (过滤窗内) -- up 单边手性通道, dn 全灭 (lead 滤波)
  注意: 对照态不是 QSH! 模型两自旋块的 (hx,hy) 同号 (无 sigma_y 翻转),
  陈数同号 (C_up=C_dn=-1), 两自旋右行通道在同一条边 (2026-09-20 Chern
  数值核实 + Ny=48 键电流 frac_top 同边实证); 主稿两处 "quantum-spin-Hall
  control" 误标已同步改为 spin-degenerate QAH control.
  每个面板附硬自检: 垂直分离集 (列 Lx//2-1 -> Lx//2 的全部 d0 键 + d2 键)
  净流量 S 必须等于 G (Landauer 守恒, 每入射模单位通量), |S| 与 G 偏差 > 1e-6 报 FAIL.

  方向约定自检 (平面波): 均匀链 H_ab = -t, psi_n = e^{ikn} 时
  J_{b->a} = 2 Im[e^{-ik(n+1)}(-t)e^{ikn}] = 2 t sin k > 0 (k>0 右行),
  即正值 = 电流从 b 流向 a; 截面 S>0 = 净向右, 应等于 +G.

几何/参数: 与 hex_ribbon_transport.py 完全同源 (hex_kw_build v0.2 双验证地基):
  Ny=160 行, Lx=60, tilt=30 meV.nm, J0=160 meV, E = gap 中心 -D*A0^2 (auto),
  干净器件 (无无序; 鲁棒性已由 fig6 覆盖, 本图只做机制).

运行 (kwant2 终端, 本目录):
  python hex_rib_bondcurrent.py                       # 正式 (~2-6 min)
  set HBC_SMOKE=1& python hex_rib_bondcurrent.py       # 秒级冒烟 (无图无 json)
env: HBC_NY=160 HBC_LX=60 HBC_A0S=0.100,0.118 HBC_T=30 HBC_J0=160 HBC_TAG=bondcur

输出 (正式):
  figs/hex_rib_bondcurrent_{TAG}.png  (300 dpi, 2x2: 行=自旋, 列=A0)
  data/hex_rib_bondcurrent_{TAG}.json (每面板 G/nmodes/截面通量/自检/边缘分布)
"""
import os
import sys
import time
import json

import numpy as np
import kwant

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_kw_build as hkb          # mats/floquet/几何常量 (v0.2 V1/V2 双验证)

# ------------------------------ 参数 (env 覆盖) -------------------------------
SMOKE = os.environ.get("HBC_SMOKE", "0") == "1"
NY = int(os.environ.get("HBC_NY", "160"))                   # 宽 (行数)
LX = int(os.environ.get("HBC_LX", "60"))                    # 散射区长 (沿 a1)
A0S = [float(x) for x in os.environ.get("HBC_A0S", "0.100,0.118").split(",")]
TILT = float(os.environ.get("HBC_T", "30.0"))               # meV.nm (单位铁律)
J0 = float(os.environ.get("HBC_J0", "160.0"))               # meV
TAG = os.environ.get("HBC_TAG", "bondcur")
HERE = os.path.dirname(os.path.abspath(__file__))

if SMOKE:                       # 秒级冒烟: 微型几何, 双 A0 双自旋, 不产任何文件
    NY = int(os.environ.get("HBC_SNY", "12"))              # 可调冒烟宽度
    LX = int(os.environ.get("HBC_SLX", "8"))
    A0S = [float(x) for x in os.environ.get(
        "HBC_SA0S", "0.100,0.118").split(",")]

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3


def log(msg):
    print(msg, flush=True)


# ---------------------------- 器件 (记录 hopping 清单) -------------------------
def make_system(Ny, Lx, spin, At, Mt, A0v):
    """同 hex_ribbon_transport.make_system, 额外返回:
       hops    = [(site_a, site_b, H_ab), ...] 与 syst[a, b] = H_ab 一一对应
       cut_idx = 垂直截面 d0 键 (列 Lx//2-1 -> Lx//2) 在 hops 中的下标."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = hkb.mats(spin, At, Mt, A0v, TILT, J0)

    def build_lead():
        lead = kwant.Builder(kwant.TranslationalSymmetry([1.0, 0.0]))
        for m in range(Ny):
            lead[lat(0, m)] = oc
        for m in range(Ny):
            lead[lat(1, m), lat(0, m)] = t0          # d0 = a1 (跨单胞)
            if m + 1 < Ny:
                lead[lat(0, m + 1), lat(0, m)] = t1      # d1 = a2
                lead[lat(-1, m + 1), lat(0, m)] = t2     # d2 = a2-a1
        return lead

    syst = kwant.Builder()
    for x in range(Lx):
        for m in range(Ny):
            syst[lat(x, m)] = oc
    hops = []
    cut_d0, cut_d2 = [], []          # 左右分离集: d0(xc->xc+1) + d2(xc+1->xc)
    xc = Lx // 2 - 1                  # 分离面所在列间隙
    for x in range(Lx):
        for m in range(Ny):
            if x + 1 < Lx:
                a, b = lat(x + 1, m), lat(x, m)
                syst[a, b] = t0
                hops.append((a, b, t0))
                if x == xc:
                    cut_d0.append(len(hops) - 1)
            if m + 1 < Ny:
                a, b = lat(x, m + 1), lat(x, m)
                syst[a, b] = t1
                hops.append((a, b, t1))
            if x >= 1 and m + 1 < Ny:
                a, b = lat(x - 1, m + 1), lat(x, m)
                syst[a, b] = t2
                hops.append((a, b, t2))
                if x == xc + 1:
                    cut_d2.append(len(hops) - 1)
    syst.attach_lead(build_lead().reversed())            # 左 lead (=lead 0)
    syst.attach_lead(build_lead())                       # 右 lead (=lead 1)
    return syst.finalized(), hops, cut_d0, cut_d2


def bond_currents(fsys, hops, psi_modes):
    """J_{b->a} = 2 Im[psi_a^dag H_ab psi_b], 逐入射模累加 (单位: 每模单位通量).
    正值 = 电流从 b 流向 a (见 docstring 平面波自检)."""
    cur = np.zeros(len(hops))
    idof = fsys.id_by_site
    for psi in psi_modes:
        for k, (a, b, tab) in enumerate(hops):
            ia = 2 * idof[a]
            ib = 2 * idof[b]
            cur[k] += 2.0 * np.imag(np.vdot(psi[ia:ia + 2], tab @ psi[ib:ib + 2]))
    return cur


def run_case(spin, A0v):
    """单个 (spin, A0): 建、G、散射态、键电流、截面守恒自检、边缘分布."""
    At, Mt = hkb.floquet(spin, A0v)
    E = -D * A0v ** 2                                     # gap 中心 (auto)
    fsys, hops, cut_d0, cut_d2 = make_system(NY, LX, spin, At, Mt, A0v)
    sm = kwant.smatrix(fsys, E)
    G = float(sm.transmission(1, 0))
    Tback = float(sm.transmission(0, 1))
    psi = np.asarray(kwant.wave_function(fsys, E)(0))     # 左 lead 入射态
    nmodes = int(psi.shape[0])
    cur = bond_currents(fsys, hops, psi)
    # 净向右通量 = 分离集求和: d0 (b=列xc -> a=列xc+1, 右向为 +)
    #              减 d2 (b=列xc+1 -> a=列xc, 左向为 +, 右向为 -)
    S = float(np.sum(cur[cut_d0]) - np.sum(cur[cut_d2]))
    if G > 1e-9:
        cut_ok = abs(abs(S) - G) < 1e-6 * max(1.0, G)
        sgn = 1.0 if S > 0 else -1.0
    else:
        cut_ok = abs(S) < 1e-6
        sgn = 1.0
    # 上下半边电流权重 (中点 y 判; 只统计 |J| > 1e-6 的活键)
    ymid = np.array([0.5 * (a.pos[1] + b.pos[1]) for a, b, _ in hops])
    act = np.abs(cur) > 1e-6
    yhalf = NY * SQ3 / 2.0 / 2.0
    if act.any():
        w = np.abs(cur[act])
        frac_top = float(np.sum(w[ymid[act] > yhalf]) / np.sum(w))
    else:
        frac_top = float("nan")
    res = {"A0": A0v, "spin": spin, "E": E, "At": At, "Mt": Mt,
           "G": G, "Tback": Tback, "nmodes_lead0": nmodes,
           "cut_flux_S": S, "cut_check_PASS": bool(cut_ok), "sign": sgn,
           "frac_current_top_half": frac_top,
           "max_abs_J": float(np.abs(cur).max())}
    log("  spin=%-2s A0=%.3f  E=%7.3f  G=%.6f  Tback=%.6f  nmodes=%d"
        "  S=%+.6f  cut=%s  frac_top=%.3f"
        % (spin, A0v, E, G, Tback, nmodes, S,
           "PASS" if cut_ok else "FAIL", frac_top))
    return res, fsys, hops, cur


def draw(panels, order_spins, order_a0, out_png):
    """2x2 面板: 行=自旋(up 上, dn 下), 列=A0(对照左, 过滤窗右).
    v1.1 抛光: 色标动态范围压到 2 个量级 (vmax*1e-2 起步, 隐藏倏逝尾弱键),
    箭头 400->150 (只标主导通道), 图面更接近论文版式."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    vmax = max(p["max_abs_J"] for p in panels.values() if p["max_abs_J"] > 0)
    vmin = vmax * 1e-2                     # v1.1: 2 个量级动态窗, 弱倏逝尾压入底色
    nrows, ncols = len(order_spins), len(order_a0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.6 * nrows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    lc_last = None
    for i, spin in enumerate(order_spins):
        for j, A0v in enumerate(order_a0):
            ax = axes[i][j]
            p = panels[(spin, A0v)]
            hops, cur = p["_hops"], p["_cur"]
            segs, vals = [], []
            for (a, b, _), v in zip(hops, cur):
                pa, pb = np.asarray(a.pos), np.asarray(b.pos)
                segs.append([pa, pb])
                vals.append(max(abs(v), vmin))
            lc = LineCollection(segs, array=np.array(vals), cmap="inferno",
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                                linewidths=0.7)
            ax.add_collection(lc)
            lc_last = lc
            # 方向箭头: 取 |J| 最强的前 150 键 (正值 = b->a) [v1.1: 400->150]
            nk = min(150, len(cur))
            for k in np.argsort(-np.abs(cur))[:nk]:
                a, b, _ = hops[k]
                if abs(cur[k]) < 0.02 * vmax:
                    break
                pa, pb = np.asarray(a.pos), np.asarray(b.pos)
                mid = 0.5 * (pa + pb)
                d = (pa - pb) if cur[k] > 0 else (pb - pa)
                d = 0.9 * d / np.linalg.norm(d)
                ax.quiver(mid[0], mid[1], d[0], d[1], color="w",
                          angles="xy", scale_units="xy", scale=1.0,
                          width=0.004, alpha=0.9)
            ax.set_xlim(-1, LX + NY / 2.0 + 1)
            ax.set_ylim(-1, NY * SQ3 / 2.0 + 1)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            lbl = "up" if spin == "up" else "dn"
            tag = "spin-degenerate QAH" if j == 0 else "single-spin filter"
            arrow = r"$\uparrow$" if spin == "up" else r"$\downarrow$"
            ax.set_title(r"(%s) spin %s, $A_0=%.3f$ nm$^{-1}$ (%s)"
                         % (chr(ord("a") + i * ncols + j), arrow, A0v, tag),
                         fontsize=11)
            gtxt = r"$G_{%s}=%.3f\,e^2/h$" % (lbl, p["G"])
            if p["G"] < 1e-9:
                gtxt += "\nno propagating modes"
            ax.text(0.03, 0.05, gtxt, transform=ax.transAxes, fontsize=10,
                    color="w", va="bottom",
                    bbox=dict(fc="black", alpha=0.55, ec="none", pad=2))
    fig.colorbar(lc_last, ax=axes, shrink=0.55, pad=0.015,
                 label=r"bond current $|I_{ij}|$ (arb. units, log scale)")
    fig.suptitle("Real-space bond currents of the hex ribbon spin filter "
                 "(injection from left lead; arrows on the 150 strongest bonds)",
                 fontsize=12)
    fig.savefig(out_png, dpi=300)
    print("[ok] %s" % out_png)


# ---------------------------------- main --------------------------------------
def main():
    t0 = time.time()
    mode = "SMOKE (Ny=%d Lx=%d, no output files)" % (NY, LX) if SMOKE \
        else "FORMAL (Ny=%d Lx=%d)" % (NY, LX)
    log("== hex ribbon bond-current figure v1.1  %s  A0S=%s  tilt=%g  J0=%g =="
        % (mode, A0S, TILT, J0))
    panels = {}
    for A0v in A0S:
        for spin in ("up", "dn"):
            res, fsys, hops, cur = run_case(spin, A0v)
            res["_hops"], res["_cur"] = hops, cur
            panels[(spin, A0v)] = res
    nfail = sum(1 for p in panels.values() if not p["cut_check_PASS"])
    log("=> %d/%d panels cut-conservation PASS" % (len(panels) - nfail, len(panels)))
    if SMOKE:
        log("[smoke done, %.1f s] no files written" % (time.time() - t0))
        return 0 if nfail == 0 else 1

    out_png = os.path.join(HERE, "figs", "hex_rib_bondcurrent_%s.png" % TAG)
    out_js = os.path.join(HERE, "data", "hex_rib_bondcurrent_%s.json" % TAG)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_js), exist_ok=True)
    draw(panels, ["up", "dn"], A0S, out_png)
    dump = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                       "tilt": TILT, "J0": J0, "Ny": NY, "Lx": LX,
                       "A0_list": A0S, "E_mode": "auto(-D*A0^2)",
                       "lattice": "triangular a=1, ribbon along a1, 2 orbs/site",
                       "convention": "J_b->a = 2 Im[psi_a^dag H_ab psi_b], "
                                     "cut flux S must equal +G"},
           "panels": {"%s@%g" % (k[0], k[1]):
                      {kk: vv for kk, vv in v.items()
                       if not kk.startswith("_")} for k, v in panels.items()}}
    with open(out_js, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    log("[ok] %s  (%.1f s)" % (out_js, time.time() - t0))
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

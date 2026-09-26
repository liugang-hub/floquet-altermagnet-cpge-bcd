# -*- coding: utf-8 -*-
r"""
hex_rib_probe.py -- 诊断: A0>=0.115 双自旋全关是"有限宽边界态杂交"吗?
========================================================================
B2 首跑现象: A0<=0.100 时 G_up=G_dn=1.000 (每自旋 1 条手性边界态, 量化);
A0>=0.115 起 G_up 也归零, 尽管 M~_up 仍为负 (A0=0.115: -1.81 meV, bulk 拓扑,
体 gap ~3.6 meV), E 也在 gap 中心. T=0 说明 E 落在 lead 的"全谱隙"里.
主要假设 (H1): 有限宽边界态杂交 -- |M~| 变小 => 边界态穿透深度
  xi ~ At/(2|M~|) 变长, 当 xi ~ Ny 时上下边缘态在 E 处反交叉打开杂交隙,
  E 钉在隙中心 => 无传播模 => T=0.  判据: 加大 Ny, lead 在 E 处应重新
  出现传播模; E 扫描应看到"死窗"随 Ny 增大而收缩.

本脚本只数 lead 传播模 (无需 smatrix, 快), 直接检验:
  Part 1  Ny 扫描: 固定 A0, 对 spin in {up,dn}, Ny in list,
          在 E_c = -D A0^2 处数 lead 传播模总数 nmodes.
          (0 => E 在全谱隙; >=2 => 边界通道存在 (每自旋 1 去 1 回); 很大 => 体态)
  Part 2  E 扫描: 固定 (A0, spin), Ny in {48, 128},
          扫 E = E_c + [-LO, +HI], 打印 nmodes 随 E 的"死窗"结构
          (死窗 = 杂交隙; 两侧 nmodes>=2 的边缘通道窗 = 可用输运能区).

几何/参数与 hex_ribbon_transport 完全同源 (import hrt.make_system 复用),
leads 干净、onsite 带 -D A0^2 (A0v 逐点传入). 不写 Paper H 官方数据.

运行 (kwant2 终端, 本目录):
  快速 (默认: 4 A0 x 2 spin x 2 Ny 数模 + E 扫描, ~1-2 min):
      python hex_rib_probe.py
  全 Ny 列表:     set PRB_NYS=48,96,128,160& python hex_rib_probe.py
  只跑 E 扫描:    set PRB_A0S=none& python hex_rib_probe.py
  E 扫描换 A0/范围: set PRB_ESC_A0=0.115& set PRB_ESC_LO=-2& set PRB_ESC_HI=2& python hex_rib_probe.py
  env: PRB_A0S=0.110,0.115,0.120,0.125  PRB_NYS=48,128  PRB_SPIN=both
       PRB_ESC_A0=0.120  PRB_ESC_SPIN=up  PRB_ESC_NY0=48  PRB_ESC_NY1=128
       PRB_ESC_LO=-3.0  PRB_ESC_HI=3.0  PRB_ESC_STEP=0.25
输出: data/hex_rib_probe.json
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_kw_build as hkb             # mats/floquet/常量 (v0.2 V1/V2 验证)
import hex_ribbon_transport as hrt     # make_system (仅复用其几何构造)

# ------------------------------ 参数 (env 覆盖) -------------------------------
_A0E = os.environ.get("PRB_A0S", "0.110,0.115,0.120,0.125")
A0S = [] if _A0E.strip().lower() == "none" else [float(x) for x in _A0E.split(",")]
NYS = [int(x) for x in os.environ.get("PRB_NYS", "48,128").split(",")]
SPIN_STR = os.environ.get("PRB_SPIN", "both").lower()
SPINS = ["up", "dn"] if SPIN_STR == "both" else [SPIN_STR]
ESC_A0 = float(os.environ.get("PRB_ESC_A0", "0.120"))
ESC_SPIN = os.environ.get("PRB_ESC_SPIN", "up").lower()
ESC_NY0 = int(os.environ.get("PRB_ESC_NY0", "48"))
ESC_NY1 = int(os.environ.get("PRB_ESC_NY1", "128"))
ESC_LO = float(os.environ.get("PRB_ESC_LO", "-3.0"))
ESC_HI = float(os.environ.get("PRB_ESC_HI", "3.0"))
ESC_STEP = float(os.environ.get("PRB_ESC_STEP", "0.25"))

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_probe.json")


def log(msg):
    print(msg, flush=True)


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (与 transport auto 同口径)."""
    return -D * A0v ** 2


def analytic(A0v):
    """解析 At_s/M~_s/Delta/xi (Floquet 公式, 打印对照用; 不参与判定)."""
    out = {}
    for s in ("up", "dn"):
        At, Mt = hkb.floquet(s, A0v)
        At, Mt = float(At), float(Mt)
        gap = 2.0 * abs(Mt)
        xi = At / gap if gap > 1e-9 else float("inf")
        dec = {Ny: float(np.exp(-Ny / xi)) if np.isfinite(xi) else 1.0
               for Ny in NYS}
        out[s] = {"At": At, "Mt": Mt, "gap2": float(gap),
                  "xi_rows": float(xi) if np.isfinite(xi) else None,
                  "e(-Ny/xi)": dec}
    return out


def lead_nmodes(fsys, E):
    """lead[0] 在 E 处的传播模总数 (0=全谱隙; 大数=体态)."""
    try:
        prop, _ = fsys.leads[0].modes(E)
        return int(len(np.asarray(prop.velocities)))
    except Exception:
        return -1


def mode_scan(A0v, Ny, spins):
    """Part 1: 对每个 spin, 数 lead 在 E_c 的传播模."""
    res = {}
    E = Ec(A0v)
    for spin in spins:
        At, Mt = hkb.floquet(spin, A0v)
        fsys = hrt.make_system(Ny, hrt.LX, spin, At, Mt, A0v)
        nm = lead_nmodes(fsys, E)
        res[spin] = {"nmodes": nm, "Mt": float(Mt)}
        log("    A0=%6.3f  Ny=%-4d  spin=%-3s  E_c=%7.3f  nmodes=%d"
            "  (M~=%+7.3f)"
            % (A0v, Ny, spin, E, nm, Mt))
    return res


def escan(A0v, spin, Ny, lo, hi, step):
    """Part 2: E 扫描 nmodes, 报告死窗边界."""
    At, Mt = hkb.floquet(spin, A0v)
    E0 = Ec(A0v)
    fsys = hrt.make_system(Ny, hrt.LX, spin, At, Mt, A0v)
    Es = np.arange(lo, hi + 0.5 * step, step)
    rows = []
    for E in Es:
        nm = lead_nmodes(fsys, E0 + E)
        rows.append((float(E0 + E), nm))
    # 死窗 = nmodes==0 的连续区
    dead = []
    run = []
    for E, nm in rows:
        if nm == 0:
            run.append(E)
        else:
            if run:
                dead.append((run[0], run[-1], len(run)))
            run = []
    if run:
        dead.append((run[0], run[-1], len(run)))
    alive = [(E, nm) for E, nm in rows if nm > 0]
    n_alive = len(alive)
    log("    [Escan] A0=%g spin=%s Ny=%d  E_c=%.3f  %d 个能量点, "
        "nmodes>0 点数=%d" % (A0v, spin, Ny, E0, len(rows), n_alive))
    for a, b, n in dead:
        log("      死窗: E in [%.3f, %.3f]  (%d 点)" % (a, b, n))
    for E, nm in alive[:6]:
        log("      活点: E=%.3f  nmodes=%d" % (E, nm))
    if n_alive > 6:
        log("      ... 共 %d 个活点 (首 6 个如上)" % n_alive)
    return {"E0": E0, "rows": rows, "dead_windows": dead}


def main():
    t0 = time.time()
    log("== hex ribbon probe: 有限宽杂交诊断  Ny=%s  A0=%s =="
        % (NYS, A0S))
    res = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                      "tilt": hrt.TILT, "J0": hrt.J0,
                      "A0_list": A0S, "Ny_list": NYS, "spins": SPINS},
           "part1": {}, "part2": {}}

    # Part 1: Ny 扫描 (数 E_c 处传播模)
    if A0S:
        log("\n[Part 1] lead 传播模 @ E_c = -D A0^2  (0 => E 在全谱隙)")
        for A0v in A0S:
            ana = analytic(A0v)
            for s in ("up", "dn"):
                a = ana[s]
                xi = a["xi_rows"]
                xis = "inf" if xi is None else "%.0f" % xi
                log("  -- A0=%.3f %s: At=%.1f M~=%+.2f (gap %.2f, xi ~ %s 行)"
                    % (A0v, s, a["At"], a["Mt"], a["gap2"], xis))
            res["part1"]["%g" % A0v] = {}
            for Ny in NYS:
                rr = mode_scan(A0v, Ny, SPINS)
                res["part1"]["%g" % A0v]["Ny%d" % Ny] = rr
    else:
        log("\n[Part 1] PRB_A0S=none, 跳过")

    # Part 2: E 扫描 (死窗随 Ny 收缩?)
    log("\n[Part 2] E 扫描 nmodes  (A0=%g spin=%s, Ny=%d 与 %d, "
        "E=E_c+[%g,%g] step %g)" % (ESC_A0, ESC_SPIN, ESC_NY0, ESC_NY1,
                                    ESC_LO, ESC_HI, ESC_STEP))
    for Ny in (ESC_NY0, ESC_NY1):
        rr = escan(ESC_A0, ESC_SPIN, Ny, ESC_LO, ESC_HI, ESC_STEP)
        res["part2"]["Ny%d" % Ny] = rr

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("\n[ok] %s  (%.1f s)" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

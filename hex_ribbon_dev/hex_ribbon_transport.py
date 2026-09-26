# -*- coding: utf-8 -*-
r"""
hex_ribbon_transport.py -- 方向 B v0.2: hex ribbon 二端自旋分辨输运 vs A0
==========================================================================
目标 (hex 版全光开关, 对照 Paper H bulk 预言):
  Paper H (bulk, hex 连续/TB): Floquet 重整化质量
      M~_s(A0) = M0 - B A0^2 - s (A0^2 A^2/hw),  s=+1 up / -1 dn, sp=+1
  dn 自旋在 A0*_dn = sqrt(|M0|/(A^2/hw - B)) ~ 0.1153 nm^-1 翻零转 trivial,
  up 自旋在 A0*_up ~ 0.1271 才翻 (Paper H: A0=0.12 时 M~_dn=+0.84 meV 已正).
  本脚本在均匀 hex ribbon (无畴壁) 上做二端输运, 把 bulk 翻转变成器件指纹:
      E 固定在 gap 中心 E_c = -D A0^2 (h0 的 k=0 常数; 两自旋共用,
      D=-512 => E_c = +512 A0^2 meV, A0=0.12 时 7.37 meV -- 不跟随会掉进价带!)
      每自旋: |C_s|=1 (M~_s<0, 拓扑) => E_c 处 1 条手性边界态 => G_s ~ 1;
              M~_s>0 (trivial)  => E_c 处无通道 => G_s ~ 0.
  预期指纹 (R1-R3, 数值由 kwant 裁决):
      R1  A0 in {0, 0.06}:  up/dn 均拓扑 (M~ = -10/-7.77/-7.29) => G ~ 2, P ~ 0
      R2  0.1153 < A0 < 0.1271:  dn 已翻 (G_dn -> 0), up 未翻 (G_up ~ 1)
            => G ~ 1, P -> +1   (单自旋过滤窗 = bulk 翻转的器件化)
      R3  A0 > 0.1271:  up 也翻 => G -> 0 (全关, P 无定义归 0)
  另: 临界点附近 (A0 略小于翻转值) gap 关闭, E_c 处体态介入 =>
      G_dn 可能出现非量化峰再塌缩, 这是"临界标记", 不是错误.

几何 (与 hex_kw_build v0.2 同一套, 已 V1/V2 双验证):
  三角格 a=1, a1=(1,0), a2=(1/2, sqrt3/2); ribbon 沿 a1, 宽 Ny 行 (m=0..Ny-1,
  zigzag 边). 散射区 Lx x Ny (斜平行四边形), 每 site 2 轨道 (单自旋块).
  三 bond (syst[site+d, site] = t, 见 hex_kw_build docstring 推导):
      t0: (x+1,m) <- (x,m)        [a1]
      t1: (x,m+1) <- (x,m)        [a2]
      t2: (x-1,m+1) <- (x,m)      [a2-a1, 只 x>=1; x=0 列左连由 lead 提供]
  leads 干净 (理想接触), 沿 a1 无限, 与散射区共享 lattice 实例与全部 hops.
  自旋分辨: 模型自旋块对角 (a1 通道), 双自旋各自独立 2x2 系统 (照 Paper F
  alm_domainwall a1 路径), G_up/G_dn = transmission(1,0).

运行 (kwant2 终端, 本目录; env 全有默认值, 每条命令显式 set 关键变量):
  主结果 (7 A0 点 x 2 自旋, ~1-3 min):
      python hex_ribbon_transport.py
  冒烟 (3 点快速看跑通):  set HRT_A0S=0,0.06,0.12& python hex_ribbon_transport.py
  宽谱扫描:              set HRT_A0S=0,0.02,0.04,0.06,0.08,0.10,0.115,0.12,0.125,0.13,0.14& python hex_ribbon_transport.py
  调宽度 (边界态杂交):   set HRT_NY=32& python hex_ribbon_transport.py
  固定 E 对照 (非跟随):  set HRT_E=0& python hex_ribbon_transport.py
  env: HRT_SPIN=both|up|dn  HRT_NY=48  HRT_LX=60  HRT_T=30(tilt meV.nm)
       HRT_J0=160  HRT_E=auto(=-D A0^2)  HRT_TAG=a0

输出: data/hex_rib_transport_{TAG}.json, rows[A0] = {E, At/Mt, G_up, G_dn, G,
      P, Tback_up, Tback_dn}   (A0 小数为键时用 %g 字符串)
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
SPIN = os.environ.get("HRT_SPIN", "both").lower()          # up | dn | both
NY = int(os.environ.get("HRT_NY", "48"))                   # 宽 (行数, ~41.6 nm @ a=1nm)
LX = int(os.environ.get("HRT_LX", "60"))                   # 散射区长 (沿 a1)
A0S = [float(x) for x in os.environ.get(
    "HRT_A0S", "0,0.06,0.10,0.115,0.12,0.13,0.14").split(",")]
TILT = float(os.environ.get("HRT_T", "30.0"))              # meV.nm (单位铁律)
J0 = float(os.environ.get("HRT_J0", "160.0"))              # meV
E_MODE = os.environ.get("HRT_E", "auto").lower()           # auto = -D A0^2
TAG = os.environ.get("HRT_TAG", "a0")

assert SPIN in ("up", "dn", "both"), "HRT_SPIN 只支持 up|dn|both"

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3

# 解析翻转 A0* (M~_s = M0 - B A0^2 - s A0^2 A^2/HW = 0 之根, sp=+1)
A0STAR = {"dn": float(np.sqrt(abs(M0) / (A ** 2 / HW - B))),       # 0.1153
          "up": float(np.sqrt(abs(M0) / (-B - A ** 2 / HW)))}      # 0.1271

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_transport_%s.json" % TAG)


def log(msg):
    print(msg, flush=True)


# ------------------------------ 器件 (每自旋 2x2) ----------------------------
def _mats(spin, At, Mt, A0v):
    """统一取 (oc,t0,t1,t2): A0v 进 onsite -D A0^2, TILT/J0 取模块值."""
    return hkb.mats(spin, At, Mt, A0v, TILT, J0)


def make_system(Ny, Lx, spin, At, Mt, A0v):
    """hex ribbon 二端器件 (无无序). 散射区 Lx 列 x Ny 行 + 双干净 lead."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = _mats(spin, At, Mt, A0v)

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
    for x in range(Lx):
        for m in range(Ny):
            if x + 1 < Lx:
                syst[lat(x + 1, m), lat(x, m)] = t0
            if m + 1 < Ny:
                syst[lat(x, m + 1), lat(x, m)] = t1
            if x >= 1 and m + 1 < Ny:
                syst[lat(x - 1, m + 1), lat(x, m)] = t2
    syst.attach_lead(build_lead().reversed())          # 左 lead (沿 -a1)
    syst.attach_lead(build_lead())                     # 右 lead (沿 +a1)
    return syst.finalized()


def energy(A0v):
    """gap 中心 E_c = -D A0^2 (h0 的 k=0 常数, 两自旋共用). E_MODE=auto 跟随."""
    if E_MODE == "auto":
        return -D * A0v ** 2
    return float(E_MODE)


def one_case(spin, A0v):
    """单个 (spin, A0): 返回 {At, Mt, E, G, P, Tback}."""
    At, Mt = hkb.floquet(spin, A0v)
    E = energy(A0v)
    fsys = make_system(NY, LX, spin, At, Mt, A0v)
    sm = kwant.smatrix(fsys, E)
    v = {"At": At, "Mt": Mt, "E": E,
         "T": float(sm.transmission(1, 0)),
         "Tback": float(sm.transmission(0, 1))}
    return v


# ---------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    spins = ["up", "dn"] if SPIN == "both" else [SPIN]
    log("== hex ribbon 二端输运 vs A0  spin=%s  Ny=%d  Lx=%d  tilt=%g  J0=%g =="
        % (SPIN, NY, LX, TILT, J0))
    log("   解析翻转 A0*:  dn=%.4f  up=%.4f   (Paper H bulk 预言)"
        % (A0STAR["dn"], A0STAR["up"]))
    log("   E 模式: %s  (auto = gap 中心 -D A0^2)" % E_MODE)
    log("输出: %s\n" % OUT)

    rows = {}
    for A0v in A0S:
        row = {"params_note": "E at gap center -D*A0^2 unless HRT_E set"}
        for spin in spins:
            v = one_case(spin, A0v)
            row[spin] = v
        if SPIN == "both":
            gu = row["up"]["T"]
            gd = row["dn"]["T"]
            g = gu + gd
            row["G_up"] = gu
            row["G_dn"] = gd
            row["G"] = g
            row["P"] = (gu - gd) / g if g > 1e-9 else 0.0
            row["Tback_up"] = row["up"]["Tback"]
            row["Tback_dn"] = row["dn"]["Tback"]
            log("  A0=%6.3f  E=%7.3f  G_up=%5.3f  G_dn=%5.3f  G=%5.3f  P=%+5.3f"
                "   (M~up=%+6.2f M~dn=%+6.2f)"
                % (A0v, row["up"]["E"], gu, gd, g,
                   row["P"], row["up"]["Mt"], row["dn"]["Mt"]))
        else:
            s = SPIN
            row["G"] = row[s]["T"]       # 单自旋模式: G 即该自旋透射
            row["P"] = 0.0               # 单自旋无极化定义
            log("  A0=%6.3f  E=%7.3f  G_%s=%5.3f  Tback=%5.3f  (M~=%+6.2f)"
                % (A0v, row[s]["E"], s, row[s]["T"], row[s]["Tback"],
                   row[s]["Mt"]))
        rows["%g" % A0v] = row

    res = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                      "tilt": TILT, "J0": J0, "Ny": NY, "Lx": LX,
                      "A0_list": A0S, "spin": SPIN, "E_mode": E_MODE,
                      "A0star_dn": A0STAR["dn"], "A0star_up": A0STAR["up"],
                      "lattice": "triangular a=1, ribbon along a1, 2 orbs/site"},
           "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("\n[ok] %s  总用时 %.1f s" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

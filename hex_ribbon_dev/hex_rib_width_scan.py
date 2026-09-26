# -*- coding: utf-8 -*-
r"""
hex_rib_width_scan.py -- B3a: 过滤窗宽度阈值标度  Ny_c(A0) ~ C * xi_up(A0)
==========================================================================
B2 收官 (ny96/128/160 x 0.115-0.120) 确立:
  * dn 关死左缘 ~A0=0.115, 无宽度依赖 (M~_dn 翻转/近闭合 => 物理关闭);
  * up 通道的"过滤窗右缘"由宽度控制: A0=0.120 时 ny96 死 -> ny128 活,
    即 up 在 E_c 处的通道阈值 Ny_c 随 A0 增大 (|M~_up| 缩小, 穿透深度
    xi_up = At/(2|M~_up|) 拉长) 而右移.
本脚本量化 Ny_c(A0), 检验标度律:
      Ny_c(A0) ~ C * xi_up(A0),   C = 器件窗需要容纳几个穿透深度.
C 若 ~2-3 且四点单调 => 有限宽过滤窗"可设计": 给定 A0 (光强) 与 Ny (宽度),
过滤窗存在性由 Ny >= C*xi_up 判定.  这是 B 线论文的器件设计标度料.

两阶段 (省时):
  Part 1  nmodes 快扫 (无 smatrix): 对 A0 in A0S, Ny in NYS, 数 up 自旋
          lead 在 E_c=-D A0^2 的传播模 (0=死, 2=活), 取首个活 Ny 为 Ny_c0;
          同时记录 dn 全程模数 (A0>=0.115 应为全 0 = 过滤窗的 dn 半边).
  Part 2  transport 精跑 (smatrix, 双自旋): 在每 A0 的 Ny_c0 邻域
          [Ny_c0-P2WIN, Ny_c0+P2WIN] 内按 P2STEP 取点, 报 G_up/G_dn/G/P,
          确认: (a) Ny_c0 处 G_up=1.000 量化; (b) G_dn=0.000 全程成立;
          (c) 阈值处 G_up 从 0 到 1 的转变区间宽度 (可对照 2|M~| gap).

几何/参数与 hex_ribbon_transport 完全同源 (import hrt.make_system), E 模式
固定 auto (E_c = -D A0^2).  不写 Paper H 官方数据.

运行 (kwant2 终端, 本目录):
  默认 (4 A0 x 13 Ny 数模 + 每 A0 5 点 transport, ~2-3 min):
      python hex_rib_width_scan.py
  只数模 (快, ~10 s):  set WR_P2=0& python hex_rib_width_scan.py
  自定 A0 / Ny 列表:  set WR_A0S=0.118,0.120& set WR_NYS=48,64,80,96,112,128& python hex_rib_width_scan.py
  env: WR_A0S=0.115,0.118,0.120,0.122  默认 NYS=32..224 step16
       WR_P2=1   WR_P2WIN=24   WR_P2STEP=12   WR_LX=60 (透射区长, 传 hrt)
输出: data/hex_rib_width_scan.json
"""
import os
import sys
import time
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_kw_build as hkb             # mats/floquet/常量 (v0.2 V1/V2 验证)
import hex_ribbon_transport as hrt     # make_system / energy / one_case 同源

# ------------------------------ 参数 (env 覆盖) -------------------------------
A0S = [float(x) for x in os.environ.get(
    "WR_A0S", "0.115,0.118,0.120,0.122").split(",")]
_NYS = os.environ.get("WR_NYS", "")
NYS = ([int(x) for x in _NYS.split(",")] if _NYS else
       list(range(32, 225, 16)))       # 32,48,...,224
DO_P2 = os.environ.get("WR_P2", "1").lower() not in ("0", "false", "no")
P2WIN = int(os.environ.get("WR_P2WIN", "24"))     # Ny_c0 邻域半宽
P2STEP = int(os.environ.get("WR_P2STEP", "12"))   # P2 采样步长
hrt.LX = int(os.environ.get("WR_LX", "60"))       # 透射区长 (hrt 模块变量)

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_width_scan.json")


def log(msg):
    print(msg, flush=True)


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (transport auto 同口径)."""
    return -D * A0v ** 2


def xi_up(A0v):
    """up 自旋边界态穿透深度粗估 xi = At/(2|M~|)  (行)."""
    At, Mt = hkb.floquet("up", A0v)
    gap = 2.0 * abs(float(Mt))
    if gap < 1e-9:
        return None
    return float(At) / gap


def lead_nmodes(fsys, E):
    """lead[0] 在 E 处传播模总数 (0=全谱隙; >=2=边界通道)."""
    try:
        prop, _ = fsys.leads[0].modes(E)
        return int(len(np.asarray(prop.velocities)))
    except Exception:
        return -1


# ---------------------------------- main ------------------------------------
def main():
    t0 = time.time()
    log("== hex ribbon 宽度阈值标度  A0=%s  Ny=%s..%s ==" % (A0S, NYS[0], NYS[-1]))
    res = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                      "tilt": hrt.TILT, "J0": hrt.J0,
                      "A0_list": A0S, "Ny_list": NYS, "Lx": hrt.LX},
           "part1": {}, "part2": {}}

    # Part 1: nmodes 快扫, 定位 up 阈值 Ny_c0 (dn 应全程 0)
    log("\n[Part 1] nmodes @ E_c: up 阈值扫描 (dn 记录对照)")
    for A0v in A0S:
        E = Ec(A0v)
        xi = xi_up(A0v)
        log("  -- A0=%.3f  E_c=%7.3f  xi_up=%s 行"
            % (A0v, E, ("inf" if xi is None else "%.0f" % xi)))
        nm_up, nm_dn, nyc0 = [], [], None
        for Ny in NYS:
            mu = md = 0
            for spin in ("up", "dn"):
                At, Mt = hkb.floquet(spin, A0v)
                fsys = hrt.make_system(Ny, hrt.LX, spin, At, Mt, A0v)
                n = lead_nmodes(fsys, E)
                if spin == "up":
                    mu = n
                else:
                    md = n
            nm_up.append(mu)
            nm_dn.append(md)
            flag = ""
            if mu >= 2 and nyc0 is None:
                nyc0 = Ny
                flag = "  <== Ny_c0 (up 首次活)"
            log("    Ny=%4d  up=%d  dn=%d%s" % (Ny, mu, md, flag))
        res["part1"]["%g" % A0v] = {"E_c": E, "xi_up_rows": xi,
                                    "Ny_c0": nyc0,
                                    "nmodes_up": nm_up, "nmodes_dn": nm_dn}
        log("    => A0=%.3f: Ny_c0=%s" % (A0v, nyc0))

    # Part 2: transport 精跑 Ny_c0 邻域 (确认量化 + dn 关闭 + 转变宽度)
    if DO_P2:
        log("\n[Part 2] transport @ E_c  Ny_c0 邻域 (步长 %d, 半宽 %d)"
            % (P2STEP, P2WIN))
        for A0v in A0S:
            nyc0 = res["part1"]["%g" % A0v]["Ny_c0"]
            if nyc0 is None:
                log("  A0=%.3f: Part1 全死, 跳过 P2" % A0v)
                res["part2"]["%g" % A0v] = {"note": "no alive Ny in scan range"}
                continue
            lo = max(NYS[0], nyc0 - P2WIN)
            hi = min(NYS[-1], nyc0 + P2WIN)
            p2rows = []
            for Ny in range(lo, hi + 1, P2STEP):
                hrt.NY = Ny                       # one_case 读模块级 NY/LX
                gu = hrt.one_case("up", A0v)["T"]   # 复用 transport one_case
                gd = hrt.one_case("dn", A0v)["T"]
                g = gu + gd
                p = (gu - gd) / g if g > 1e-9 else 0.0
                p2rows.append({"Ny": Ny, "G_up": float(gu), "G_dn": float(gd),
                               "G": float(g), "P": float(p)})
                log("    A0=%.3f  Ny=%4d  G_up=%5.3f  G_dn=%5.3f  G=%5.3f"
                    "  P=%+5.3f" % (A0v, Ny, gu, gd, g, p))
            res["part2"]["%g" % A0v] = p2rows

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log("\n[ok] %s  (%.1f s)" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

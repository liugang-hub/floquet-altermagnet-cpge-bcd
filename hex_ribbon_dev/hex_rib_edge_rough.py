# -*- coding: utf-8 -*-
r"""
hex_rib_edge_rough.py -- 宁波基金补算 P1: 过滤窗 边界粗糙度 (edge roughness) 统计
================================================================================
物理 (B3c/退相干收官后的最后一个无序类型, 补报告管线(5)明文缺口:
"边界粗糙度、临界宽度的温度依赖以及多端构型尚未检验"):
  此前四类无序 (a1 sigma0 / mz sigma_z / sx sigma_x / 退相干 gamma) 全部只扰动
  onsite, 不动几何. 真实刻蚀器件的边界是粗糙的 => 边界粗糙度 = 散射区边界行
  (m < depth 或 m >= Ny-depth) 以概率 p 随机删点 (结构无序, spin 无关).
  机理预期 (写在此处待数据裁决):
    * dn 半边 G=0 仍由 lead 滤波结构保证 (B3b D2): 粗糙度无法创造 dn 入射通道
      => P=+1 结构性保持;
    * up 手性通道 (C=1) 禁背散射, 粗糙度只让通道沿边界蜿蜒, 两端透射仍为 1
      => 预期 G_up 对边界粗糙度同样免疫 (拓扑保护的最强检验);
    * 唯一失效模式: 粗糙度深到把边界行整条剥掉 (depth 大 + p 大) 迫使通道
      深入体内, 通道蜿蜒长度增加但拓扑透射不变 -- 若 G 退化, 就是物理新结论.

口径 (与 hex_rib_disorder_dp.py 逐条对齐, 只换无序类型):
  * 粗糙度只进散射区 (删点), leads 干净完整 Ny;
  * up/dn 共享同一删点集合 (结构无序, 保 s_z);
  * p 不进 seed: 每个 conf 只抽一次 u(x,m) ~ U(0,1), 删点 = (边界行) and (u < p)
    => p 嵌套 (p=0.1 的删点集合是 p=0.2 的子集), A0 间共享同一边界实现
    (几何 seed 只含 conf, 不含 A0/depth/p => 跨 A0 配对比较);
  * depth 同样嵌套: depth=2 的删点集合包含 depth=1 的;
  * p=0 行 = 干净器件, 数值必须与 hex_rib_disorder_dp 的 (W=0, gamma=0) 行
    逐位一致 (回归自检, 无随机性);
  * E = auto = -D A0^2 (gap 中心, 同口径).

网格 (默认, env 覆盖):
  A0 = 0.100 (QSH 对照, G=2, P=0) / 0.115, 0.118, 0.120 (过滤窗三点);
  p = 0, 0.05, 0.10, 0.20, 0.30, 0.50 (每边界行删点概率);
  depth = 1, 2 (粗糙深度, 单位: 边界行数);
  NC = 8 (快档) / 20 (正式).

运行 (kwant2 终端, 本目录):
  冒烟 (代码路径, ~2 min):
  set HRB_A0S=0.118& set HRB_NY=160& set HRB_LX=30& set HRB_P=0,0.2& set HRB_DEPTH=1& set HRB_NC=2& set HRB_TAG=ersmoke& python hex_rib_edge_rough.py
  快档 (4 A0 x 6 p x 2 depth x NC=8 = 384 conf, ~20-40 min):
  set HRB_A0S=0.100,0.115,0.118,0.120& set HRB_NY=160& set HRB_LX=60& set HRB_P=0,0.05,0.1,0.2,0.3,0.5& set HRB_DEPTH=1,2& set HRB_NC=8& set HRB_TAG=er_fast& python hex_rib_edge_rough.py
  正式 (NC=20 = 960 conf, ~1.5-2 h):
  set HRB_A0S=0.100,0.115,0.118,0.120& set HRB_NY=160& set HRB_LX=60& set HRB_P=0,0.05,0.1,0.2,0.3,0.5& set HRB_DEPTH=1,2& set HRB_NC=20& set HRB_TAG=er& python hex_rib_edge_rough.py
env: HRB_A0S / HRB_NY / HRB_LX / HRB_P / HRB_DEPTH / HRB_NC / HRB_TAG / HRB_NP

输出: data/hex_rib_edge_rough_{TAG}.json
  results[A0][depth][p] = 统计块 {G_avg, G_std, Tup_avg, Tdn_avg, P_avg, P_std,
  p_blk, p_flt, n_rm_avg, *_conf} (增量落盘, 随时可收割)

*** 铁律: 跑的过程绝对禁止 Ctrl+C (mp.Pool 只补 worker 不补任务, 会永久卡死).
    需要中断: taskkill /F /IM python.exe. 已完成 (A0,depth,p) 组有增量落盘. ***
"""
import os
import sys
import time
import json
import multiprocessing as mp

import numpy as np
import kwant

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hex_kw_build as hkb             # mats/floquet/常量 (v0.2 V1/V2 验证)
import hex_ribbon_transport as hrt     # TILT/J0/energy/几何 (单一来源)

# ------------------------------ 参数 (env 覆盖) -------------------------------
A0S = [float(x) for x in os.environ.get(
    "HRB_A0S", "0.118").split(",")]
NY = int(os.environ.get("HRB_NY", "160"))
LX = int(os.environ.get("HRB_LX", "60"))
P_LIST = [float(x) for x in os.environ.get(
    "HRB_P", "0,0.05,0.1,0.2,0.3,0.5").split(",")]
DEPTH_LIST = [int(x) for x in os.environ.get(
    "HRB_DEPTH", "1").split(",")]
N_CONF = int(os.environ.get("HRB_NC", "8"))
TAG = os.environ.get("HRB_TAG", "er")
NPROC = int(os.environ.get("HRB_NP", "0")) or min(os.cpu_count() or 1, 8)

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3
TILT = hrt.TILT
J0 = hrt.J0

assert min(P_LIST) >= 0.0 and max(P_LIST) <= 1.0, "p 必须在 [0,1]"
assert min(DEPTH_LIST) >= 1 and max(DEPTH_LIST) * 2 < NY, "depth 太深 (吃掉整个条带)"
assert sorted(DEPTH_LIST) == DEPTH_LIST, "DEPTH 必须升序 (嵌套口径)"
assert sorted(P_LIST) == P_LIST, "P 必须升序 (嵌套口径)"

MODE_ID = 9        # seed 空间与 B2/B3a/B3b/dp (7/8) 隔离
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_edge_rough_%s.json" % TAG)


def log(msg):
    print(msg, flush=True)


def geom_seed(c):
    """几何 seed 只含 conf: A0/depth/p 全部不进 seed =>
    各 A0 共享同一批边界实现; p/depth 靠 u 阈值嵌套."""
    return MODE_ID * 10**16 + c


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (hrt.energy auto 同口径)."""
    return -D * A0v ** 2


def make_system(Ny, Lx, spin, At, Mt, A0v, removed):
    """hex ribbon 二端器件 + 散射区边界删点 (leads 干净完整 Ny).
    removed: set of (x, m) 待删格点; hoppings 双端任一在 removed 即跳过."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = hkb.mats(spin, At, Mt, A0v, TILT, J0)

    def build_lead():
        lead = kwant.Builder(kwant.TranslationalSymmetry([1.0, 0.0]))
        for m in range(Ny):
            lead[lat(0, m)] = oc
        for m in range(Ny):
            lead[lat(1, m), lat(0, m)] = t0
            if m + 1 < Ny:
                lead[lat(0, m + 1), lat(0, m)] = t1
                lead[lat(-1, m + 1), lat(0, m)] = t2
        return lead

    syst = kwant.Builder()
    for x in range(Lx):
        for m in range(Ny):
            if (x, m) in removed:
                continue
            syst[lat(x, m)] = oc
    for x in range(Lx):
        for m in range(Ny):
            if (x, m) in removed:
                continue
            if x + 1 < Lx and (x + 1, m) not in removed:
                syst[lat(x + 1, m), lat(x, m)] = t0
            if m + 1 < Ny and (x, m + 1) not in removed:
                syst[lat(x, m + 1), lat(x, m)] = t1
            if x >= 1 and m + 1 < Ny and (x - 1, m + 1) not in removed:
                syst[lat(x - 1, m + 1), lat(x, m)] = t2
    syst.attach_lead(build_lead().reversed())          # 左 lead (沿 -a1)
    syst.attach_lead(build_lead())                     # 右 lead (沿 +a1)
    return syst.finalized()


def one_conf(A0v, depth, pv, c):
    """一个 (A0, depth, p, conf): 同一 u 抽样 => p/depth 嵌套, A0 间配对."""
    rng = np.random.default_rng(geom_seed(c))
    umap = {(x, m): float(rng.uniform(0.0, 1.0))
            for x in range(LX) for m in range(NY)}
    removed = set()
    for (x, m), u in umap.items():
        if m < depth or m >= NY - depth:
            if u < pv:
                removed.add((x, m))
    n_rm = len(removed)
    E = Ec(A0v)
    v = {}
    for spin in ("up", "dn"):
        At, Mt = hkb.floquet(spin, A0v)
        fsys = make_system(NY, LX, spin, At, Mt, A0v, removed)
        v["T_%s" % spin] = float(kwant.smatrix(fsys, E).transmission(1, 0))
    v["n_rm"] = n_rm
    return v


def _task(args):
    A0v, depth, pv, c = args
    return one_conf(A0v, depth, pv, c)


def agg_rows(cfgs):
    """cfgs: [{T_up, T_dn, n_rm}...] -> 统计块 (照 Paper F 口径)."""
    gu = np.array([cf["T_up"] for cf in cfgs])
    gd = np.array([cf["T_dn"] for cf in cfgs])
    nr = np.array([cf["n_rm"] for cf in cfgs], dtype=float)
    g = gu + gd
    p = np.where(g > 1e-9, (gu - gd) / np.maximum(g, 1e-12), 0.0)
    return {
        "G_avg": round(float(g.mean()), 6),
        "G_std": round(float(g.std()), 6),
        "Tup_avg": round(float(gu.mean()), 6),
        "Tdn_avg": round(float(gd.mean()), 6),
        "P_avg": round(float(p.mean()), 6),
        "P_std": round(float(p.std()), 6),
        "p_blk": round(float((g < 0.25).mean()), 4),
        "p_flt": round(float(((g > 0.75) & (np.abs(p) > 0.9)).mean()), 4),
        "n_rm_avg": round(float(nr.mean()), 2),
        "G_conf": [round(float(x), 6) for x in g],
        "P_conf": [round(float(x), 6) for x in p],
        "Tup_conf": [round(float(x), 6) for x in gu],
        "Tdn_conf": [round(float(x), 6) for x in gd],
    }


def dump(results):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                               "tilt": TILT, "J0": J0, "Ny": NY, "Lx": LX,
                               "A0_list": A0S, "p_list": P_LIST,
                               "depth_list": DEPTH_LIST, "N_conf": N_CONF,
                               "E_note": "auto = -D*A0^2 (gap center)",
                               "disorder": "edge roughness: boundary rows "
                                           "(m<depth or m>=Ny-depth) random "
                                           "site removal, prob p; p/depth "
                                           "nested via shared u draws; same "
                                           "geometry across A0 (seed=conf "
                                           "only)"},
                   "results": results},
                  f, ensure_ascii=False, indent=1)


def main():
    t0 = time.time()
    log("== hex ribbon 边界粗糙度统计  A0=%s  Ny=%d  Lx=%d  p=%s  depth=%s  "
        "NC=%d  NPROC=%d ==" % (A0S, NY, LX, P_LIST, DEPTH_LIST,
                                N_CONF, NPROC))
    log("  E = auto (-D A0^2); 删点只进散射区边界行, leads 干净; "
        "p/depth 嵌套 (同 u 阈值), A0 间共享边界实现")
    log("  p=0 行 = 干净器件, 须与 hex_rib_disorder_dp 的 (W=0,gamma=0) 行逐位一致")
    log("输出: %s\n" % OUT)

    pool = mp.Pool(NPROC)
    results = {}
    try:
        for A0v in A0S:
            for depth in DEPTH_LIST:
                for pv in P_LIST:
                    tasks = [(A0v, depth, pv, c) for c in range(N_CONF)]
                    cfgs = []
                    t1 = time.time()
                    for i, out in enumerate(
                            pool.imap(_task, tasks, chunksize=1), 1):
                        cfgs.append(out)
                        if i % 8 == 0 or i == N_CONF:
                            log("    [A0=%g depth=%d p=%.2f  c=%3d/%d  "
                                "本组 %5.1f s  累计 %6.0f s]"
                                % (A0v, depth, pv, i, N_CONF,
                                   time.time() - t1, time.time() - t0))
                    agg = agg_rows(cfgs)
                    results.setdefault("%g" % A0v, {}) \
                            .setdefault(str(depth), {})["%g" % pv] = agg
                    dump(results)      # 逐 (A0,depth,p) 组落盘
                    log("  [A0=%g depth=%d p=%.2f]  <G>=%.4f±%.4f  <P>=%+.4f  "
                        "Tup=%.4f Tdn=%.4f  n_rm=%.1f  p_blk=%.2f p_flt=%.2f  "
                        "[已落盘]"
                        % (A0v, depth, pv, agg["G_avg"], agg["G_std"],
                           agg["P_avg"], agg["Tup_avg"], agg["Tdn_avg"],
                           agg["n_rm_avg"], agg["p_blk"], agg["p_flt"]))
    finally:
        pool.close()
        pool.join()
    log("\n[落盘] %s  总用时 %.1f s" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
r"""
hex_rib_disorder.py -- B3b: 过滤窗无序存活统计 (Anderson 电荷无序, 口径照 Paper F)
==============================================================================
物理问题 (B3a 收官后的自然下一步):
  过滤窗 (A0 in (0.1153, A0_R(Ny)), 器件化单自旋滤波器: G_up=1, G_dn=0, P=+1)
  对杂质/无序有多鲁棒?  两个候选破坏机制:
    (i)  up 手性通道: C=1 边界态, 无序下禁止背散射 (手性保护) =>
          预期 G_up 在 W 远小于 up 体隙 (2|M~_up|) 时钉死 1.000;
          直到 W ~ up 体隙量级才可能出现退化/漏通道.
    (ii) dn 半边: trivial (M~_dn>0), E_c 钉在 dn 体隙中心 (gap=2|M~_dn|),
          无序把带尾拉进 gap => G_dn 从 0 抬头 => P 从 +1 跌落.
          容差标度候选: W_tol ~ dn gap(2|M~_dn|)  (0.118 时 0.96 meV 最脆,
          0.120 时 1.68 meV, 越深入窗内越稳 => "窗中心选点"设计规则).
  对照: A0=0.100 (QSH 区 G=2, P=0, 双拓扑通道) 在相同无序下应保持 G=2
        (helical 通道对非磁性电荷无序的 TRS/拓扑保护), 直到 W~min gap.

判据 (用户跑后裁决):
  D1  G_up(W): A0=0.115/0.118/0.120 在 Ny=160 保持 1.000 的平台宽度;
  D2  G_dn(W): 抬头位置 (G_dn>0.05 的 W 阈值) 是否 ~ dn gap;
       P(W):   从 +1 跌落的 W 阈值 => 过滤窗无序容差;
  D3  对照 A0=0.100: G=2 平台保持到 W ~ ? (预期远大于 dn 关闭的 0.118).

口径 (与 Paper F alm_domainwall a1 完全一致, 2026-08-31):
  * 无序只进散射区: onsite += W * v * DM,  v ~ U(-0.5,0.5);
    DM = sigma0 (电荷无序, HRD_MODE=a1, 默认) | sigma_z (质量无序, =mz).
  * leads 始终干净 (理想接触).
  * up/dn 各自独立 2x2 系统 (模型自旋块对角), 共享同一 vmap (同一样品),
    s_z 严格守恒 (a1/mz 均不翻转自旋) => G_dn=0/P=+1 的破坏只能来自
    dn 体态渗流, 不能来自自旋翻转 (那是后续 4 轨道路径的事).
  * E 固定 auto = -D A0^2 (gap 中心, 与 transport/width_scan 同口径).
  * 统计: NC 个独立 realization (确定性 seed, 逐位可复现), 报
    <G>±std, <P>, p_blk (G<0.25 比例), p_flt (G>0.75 且 |P|>0.9 比例).

几何/参数与 hex_ribbon_transport 同源 (import hrt 取 TILT/J0/energy),
系统构建照 hrt.make_system 逐 hop 复制 + 散射区 onsite 注入无序.

运行 (kwant2 终端, 本目录; env 全有默认, 每条命令显式 set 关键变量):
  冒烟 (1 点 x 3 W x NC=8, ~1 min):  python hex_rib_disorder.py
  正式 (推荐, 4 点 x 8 W x NC=40, ~10-20 min 视核数):
      set HRD_A0S=0.100,0.115,0.118,0.120& set HRD_NY=160& set HRD_LX=60& set HRD_W=0,0.25,0.5,1,2,4,8,16& set HRD_NC=40& set HRD_TAG=dis& python hex_rib_disorder.py
  质量无序对照 (sigma_z):  上面命令加 set HRD_MODE=mz& set HRD_TAG=dismz&
  env: HRD_A0S=0.118   HRD_NY=160   HRD_LX=60   HRD_W=0,0.5,1,2,4,8
       HRD_NC=8   HRD_NP=(自动 min(cpu,8))   HRD_MODE=a1|mz   HRD_TAG=dis
输出: data/hex_rib_disorder_{TAG}.json, results[A0][W] = 统计块 + 全 conf 列表

*** 铁律: 跑的过程绝对禁止 Ctrl+C (mp.Pool 只补 worker 不补任务, 会永久卡死).
    需要中断: taskkill /F /IM python.exe. 已完成 (A0,W) 组有增量落盘. ***
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
    "HRD_A0S", "0.118").split(",")]
NY = int(os.environ.get("HRD_NY", "160"))
LX = int(os.environ.get("HRD_LX", "60"))
W_LIST = [float(x) for x in os.environ.get(
    "HRD_W", "0,0.5,1,2,4,8").split(",")]
N_CONF = int(os.environ.get("HRD_NC", "8"))
MODE = os.environ.get("HRD_MODE", "a1").lower()   # a1 电荷 | mz 质量
TAG = os.environ.get("HRD_TAG", "dis")
NPROC = int(os.environ.get("HRD_NP", "0")) or min(os.cpu_count() or 1, 8)

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3
TILT = hrt.TILT
J0 = hrt.J0

assert MODE in ("a1", "mz"), "HRD_MODE 只支持 a1|mz"
MODE_ID = 7 + {"a1": 0, "mz": 1}[MODE]     # seed 空间与 B2/B3a 隔离

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_disorder_%s.json" % TAG)

S0 = np.eye(2, dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def log(msg):
    print(msg, flush=True)


def seed_of(A0v, Wd, c):
    """确定性 seed: (模式, A0, 无序强度, 配置) 唯一. A0*1e5, W*1e3 取整拼接."""
    a = int(round(A0v * 1e5))            # 5 位
    w = int(round(Wd * 1e3))             # 5 位 (meV*1000)
    return (MODE_ID * 10**16 + a * 10**8 + w * 10**3 + c)


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (hrt.energy auto 同口径)."""
    return -D * A0v ** 2


def make_system(Ny, Lx, spin, At, Mt, A0v, vmap, Wd):
    """hex ribbon 二端器件 + 散射区无序 (leads 干净). 几何照 hrt.make_system."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = hkb.mats(spin, At, Mt, A0v, TILT, J0)
    DM = S0 if MODE == "a1" else SZ

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

    def onsite(site):
        return oc + Wd * vmap.get(tuple(int(v) for v in site.tag), 0.0) * DM

    syst = kwant.Builder()
    for x in range(Lx):
        for m in range(Ny):
            syst[lat(x, m)] = onsite
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


def one_conf(A0v, Wd, c):
    """一个 (A0, W, conf): up/dn 共享同一 vmap, 返回 {T_up, T_dn}."""
    rng = np.random.default_rng(seed_of(A0v, Wd, c))
    vmap = {(x, m): float(rng.uniform(-0.5, 0.5))
            for x in range(LX) for m in range(NY)}
    E = Ec(A0v)
    v = {}
    for spin in ("up", "dn"):
        At, Mt = hkb.floquet(spin, A0v)
        fsys = make_system(NY, LX, spin, At, Mt, A0v, vmap, Wd)
        v["T_%s" % spin] = float(kwant.smatrix(fsys, E).transmission(1, 0))
    return v


def _task(args):
    A0v, Wd, c = args
    return one_conf(A0v, Wd, c)


def agg_rows(cfgs):
    """cfgs: [{T_up, T_dn}...] -> 统计块 (照 Paper F 口径)."""
    gu = np.array([cf["T_up"] for cf in cfgs])
    gd = np.array([cf["T_dn"] for cf in cfgs])
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
        "G_conf": [round(float(x), 6) for x in g],
        "P_conf": [round(float(x), 6) for x in p],
        "Tup_conf": [round(float(x), 6) for x in gu],
        "Tdn_conf": [round(float(x), 6) for x in gd],
    }


def dump(results):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                               "tilt": TILT, "J0": J0, "Ny": NY, "Lx": LX,
                               "A0_list": A0S, "W_list": W_LIST,
                               "N_conf": N_CONF, "mode": MODE,
                               "E_note": "auto = -D*A0^2 (gap center)"},
                   "results": results},
                  f, ensure_ascii=False, indent=1)


def main():
    t0 = time.time()
    log("== hex ribbon 无序存活  A0=%s  Ny=%d  Lx=%d  W=%s  NC=%d  mode=%s  NPROC=%d =="
        % (A0S, NY, LX, W_LIST, N_CONF, MODE, NPROC))
    log("  E = auto (-D A0^2, gap 中心); 无序只进散射区, leads 干净")
    log("输出: %s\n" % OUT)

    pool = mp.Pool(NPROC)
    results = {}
    try:
        for A0v in A0S:
            for Wd in W_LIST:
                tasks = [(A0v, Wd, c) for c in range(N_CONF)]
                cfgs = []
                t1 = time.time()
                for i, out in enumerate(
                        pool.imap(_task, tasks, chunksize=1), 1):
                    cfgs.append(out)
                    if i % 8 == 0 or i == N_CONF:
                        log("    [A0=%g W=%g  c=%3d/%d  本组 %5.1f s  累计 %6.0f s]"
                            % (A0v, Wd, i, N_CONF, time.time() - t1,
                               time.time() - t0))
                agg = agg_rows(cfgs)
                results.setdefault("%g" % A0v, {})["%g" % Wd] = agg
                dump(results)      # 逐 (A0,W) 组落盘, 随时可收割
                log("  [A0=%g W=%g]  <G>=%.4f±%.4f  <P>=%+.4f  "
                    "Tup=%.4f Tdn=%.4f  p_blk=%.2f p_flt=%.2f  [已落盘]"
                    % (A0v, Wd, agg["G_avg"], agg["G_std"], agg["P_avg"],
                       agg["Tup_avg"], agg["Tdn_avg"],
                       agg["p_blk"], agg["p_flt"]))
    finally:
        pool.close()
        pool.join()
    log("\n[落盘] %s  总用时 %.1f s" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

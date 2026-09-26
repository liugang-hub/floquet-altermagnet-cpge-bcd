# -*- coding: utf-8 -*-
r"""
hex_rib_disorder_dp.py -- 宁波基金补算 ③: 过滤窗 无序 x 退相干 联合扫描
=============================================================================
物理 (B3b/B3c 收官后的鲁棒性升级):
  过滤窗 (P=+1.000 单自旋滤波器) 已有三层无序存活证据 (a1 sigma0 / mz sigma_z /
  sx sigma_x 到 W=16 meV 退化 <0.5%, 见 story_hex_ribbon.md 4/5 节)。真实器件还
  受环境相位破碎 (声子/光子散射) 作用 => 补"退相干"维度做 (W, gamma) 联合扫描:
    * 无序 W: Anderson 电荷无序 (a1, 口径同 hex_rib_disorder);
    * 退相干 gamma: 虚导线弱耦合吸收极限, 散射区 onsite -= i*(gamma/2)*sigma0
      (相位破碎速率 gamma, 电流不守恒口径 => 给出鲁棒性的保守估计);
    * 联合判据: G_up(W,gamma) 平台跌落位置 / P 保持区间 -> 器件对
      "杂质 + 动态环境" 双重扰动的容差曲面 (研究内容一 器件化证据).
  机理预期: up 手性通道 (C=1) 禁背散射, 即使有相位破碎也只引入寿命展宽,
  平台应保持到 gamma ~ up gap (2.16-3.62 meV) 量级; dn 半边 G=0 由 lead 滤波
  结构保证 (B3b D2), 相位破碎无法创造 dn 入射通道 => P=+1 结构鲁棒.

口径 (与 hex_rib_disorder.py 逐条一致, 只增 gamma):
  * 无序只进散射区 (onsite += W*v*DM), leads 干净; v ~ U(-0.5,0.5);
  * up/dn 共享同一 vmap (同一样品, 保 s_z);
  * gamma 不进 seed: 同一 (A0,W,conf) 的不同 gamma 共享同一无序实现
    => 同一样品上加退相干的直接对比; gamma=0 行与 hex_rib_disorder_dis.json
    (同 A0,W,NC) 逐位一致 (回归自检);
  * E = auto = -D A0^2 (gap 中心); NC 个 realization 确定性 seed.

运行 (kwant2 终端, 本目录; 每条命令显式 set 关键变量):
  冒烟 (代码路径, ~1 min):  python hex_rib_disorder_dp.py
  快档看趋势 (4 A0 x 3 W x 5 gamma x NC=8 = 480 conf, ~20-40 min):
  set HRD_A0S=0.100,0.115,0.118,0.120& set HRD_NY=160& set HRD_LX=60& set HRD_W=0,8,16& set HRD_GAMMA=0,0.25,0.5,1,2& set HRD_NC=8& set HRD_TAG=disdp_fast& python hex_rib_disorder_dp.py
  正式 (NC=20 = 1200 conf, ~1.5-2 h, 建议快档确认趋势后跑):
  set HRD_A0S=0.100,0.115,0.118,0.120& set HRD_NY=160& set HRD_LX=60& set HRD_W=0,2,4,8,16& set HRD_GAMMA=0,0.125,0.25,0.5,1,2& set HRD_NC=20& set HRD_TAG=disdp& python hex_rib_disorder_dp.py
env: HRD_A0S / HRD_NY / HRD_LX / HRD_W (meV) / HRD_GAMMA (meV) / HRD_NC /
     HRD_MODE=a1|mz / HRD_TAG / HRD_NP (自动 min(cpu,8))

输出: data/hex_rib_disorder_dp_{TAG}.json
  results[A0][W][gamma] = 统计块 {G_avg, G_std, Tup_avg, Tdn_avg, P_avg,
  P_std, p_blk, p_flt, *_conf} (增量落盘, 随时可收割)

*** 铁律: 跑的过程绝对禁止 Ctrl+C (mp.Pool 只补 worker 不补任务, 会永久卡死).
    需要中断: taskkill /F /IM python.exe. 已完成 (A0,W,gamma) 组有增量落盘. ***
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
GAMMA_LIST = [float(x) for x in os.environ.get(
    "HRD_GAMMA", "0,0.25,0.5,1,2").split(",")]
N_CONF = int(os.environ.get("HRD_NC", "8"))
MODE = os.environ.get("HRD_MODE", "a1").lower()   # a1 电荷 | mz 质量
TAG = os.environ.get("HRD_TAG", "disdp")
NPROC = int(os.environ.get("HRD_NP", "0")) or min(os.cpu_count() or 1, 8)

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3
TILT = hrt.TILT
J0 = hrt.J0

assert MODE in ("a1", "mz"), "HRD_MODE 只支持 a1|mz"
assert min(GAMMA_LIST) >= 0.0, "gamma 必须非负"
MODE_ID = 7 + {"a1": 0, "mz": 1}[MODE]     # seed 空间与 B2/B3a/B3b 隔离

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_disorder_dp_%s.json" % TAG)

S0 = np.eye(2, dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def log(msg):
    print(msg, flush=True)


def seed_of(A0v, Wd, c):
    """确定性 seed: 与 hex_rib_disorder.py 完全相同 (gamma 不进 seed).
    => gamma=0 行逐位回归 B3b; 同一 (A0,W,conf) 各 gamma 共享同一样品."""
    a = int(round(A0v * 1e5))            # 5 位
    w = int(round(Wd * 1e3))             # 5 位 (meV*1000)
    return (MODE_ID * 10**16 + a * 10**8 + w * 10**3 + c)


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (hrt.energy auto 同口径)."""
    return -D * A0v ** 2


def make_system(Ny, Lx, spin, At, Mt, A0v, vmap, Wd, gam):
    """hex ribbon 二端器件 + 散射区无序 W + 退相干 gamma (leads 干净).
    几何照 hrt.make_system / hex_rib_disorder.make_system."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = hkb.mats(spin, At, Mt, A0v, TILT, J0)
    DM = S0 if MODE == "a1" else SZ
    iG = 0.5j * gam * S0                    # 虚导线弱耦合吸收 (相位破碎速率 gam)

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
        return oc + Wd * vmap.get(tuple(int(v) for v in site.tag),
                                  0.0) * DM - iG

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


def one_conf(A0v, Wd, gam, c):
    """一个 (A0, W, gamma, conf): up/dn 共享同一 vmap, 返回 {T_up, T_dn}."""
    rng = np.random.default_rng(seed_of(A0v, Wd, c))
    vmap = {(x, m): float(rng.uniform(-0.5, 0.5))
            for x in range(LX) for m in range(NY)}
    E = Ec(A0v)
    v = {}
    for spin in ("up", "dn"):
        At, Mt = hkb.floquet(spin, A0v)
        fsys = make_system(NY, LX, spin, At, Mt, A0v, vmap, Wd, gam)
        # 退相干项使散射区非厄米 (leads 干净仍厄米) => 关闭 Hermiticity 检查
        v["T_%s" % spin] = float(kwant.smatrix(
            fsys, E, check_hermiticity=False).transmission(1, 0))
    return v


def _task(args):
    A0v, Wd, gam, c = args
    return one_conf(A0v, Wd, gam, c)


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
                               "gamma_list": GAMMA_LIST, "N_conf": N_CONF,
                               "mode": MODE,
                               "E_note": "auto = -D*A0^2 (gap center)",
                               "dephasing": "virtual-lead weak-coupling "
                                            "absorption: onsite -= i*(gamma/2)*s0; "
                                            "gamma not in seed (same sample "
                                            "across gamma)"},
                   "results": results},
                  f, ensure_ascii=False, indent=1)


def main():
    t0 = time.time()
    log("== hex ribbon 无序 x 退相干联合扫描  A0=%s  Ny=%d  Lx=%d  "
        "W=%s  gamma=%s  NC=%d  mode=%s  NPROC=%d =="
        % (A0S, NY, LX, W_LIST, GAMMA_LIST, N_CONF, MODE, NPROC))
    log("  E = auto (-D A0^2, gap 中心); 无序+退相干只进散射区, leads 干净")
    log("  gamma 不进 seed: 同 (A0,W,conf) 各 gamma 共享同一无序样品; "
        "gamma=0 行逐位回归 hex_rib_disorder_dis.json")
    log("输出: %s\n" % OUT)

    pool = mp.Pool(NPROC)
    results = {}
    try:
        for A0v in A0S:
            for Wd in W_LIST:
                for gam in GAMMA_LIST:
                    tasks = [(A0v, Wd, gam, c) for c in range(N_CONF)]
                    cfgs = []
                    t1 = time.time()
                    for i, out in enumerate(
                            pool.imap(_task, tasks, chunksize=1), 1):
                        cfgs.append(out)
                        if i % 8 == 0 or i == N_CONF:
                            log("    [A0=%g W=%g gamma=%g  c=%3d/%d  "
                                "本组 %5.1f s  累计 %6.0f s]"
                                % (A0v, Wd, gam, i, N_CONF,
                                   time.time() - t1, time.time() - t0))
                    agg = agg_rows(cfgs)
                    results.setdefault("%g" % A0v, {}) \
                            .setdefault("%g" % Wd, {})["%g" % gam] = agg
                    dump(results)      # 逐 (A0,W,gamma) 组落盘, 随时可收割
                    log("  [A0=%g W=%g gamma=%g]  <G>=%.4f±%.4f  <P>=%+.4f  "
                        "Tup=%.4f Tdn=%.4f  p_blk=%.2f p_flt=%.2f  [已落盘]"
                        % (A0v, Wd, gam, agg["G_avg"], agg["G_std"],
                           agg["P_avg"], agg["Tup_avg"], agg["Tdn_avg"],
                           agg["p_blk"], agg["p_flt"]))
    finally:
        pool.close()
        pool.join()
    log("\n[落盘] %s  总用时 %.1f s" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

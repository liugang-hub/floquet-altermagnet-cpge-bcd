# -*- coding: utf-8 -*-
r"""
hex_rib_disorder_sx.py -- B3c: 自旋混合(磁性)无序最坏情况对照 (4x4, sigma_x)
==============================================================================
物理动机 (B3b mode=a1 收官后, 09-03):
  B3b 已证: 标量电荷无序 (W v sigma0) 下过滤窗 P=+1.0000 全程保持 --
  up 手性通道免疫 (C=1 无背散射), dn 侧 G_dn==0 由 lead 滤波保证
  (E_c 钉在 dn trivial gap 内 => dn lead 无传播模, 与散射区无序无关).

  但 sigma0/sigma_z 都保 s_z (自旋块对角, 不混合 up/dn). 真正的"最坏情况"
  是自旋混合 (磁性) 无序: onsite += W v (sigma_x (x) I2), 把 up 入射散射进
  dn 块. 本脚本把它做成 4x4 全系统 (up/dn 两块经 sigma_x 耦合) 检验:
    - dn lead 在 E_c 仍无传播模 => dn 出射恒 0 => 出射侧自旋极化仍 +1
      (器件级 P_out=+1 由结构保证, 不随磁性无序崩);
    - 但 up 入射经 sigma_x 损失进 dn evanescent 通道 => G_up 可下降.
      磁性无序局部关 gap (Zeeman 型) 还可能开体泄漏.
    判据:
    S1  G_up4(W) 退化阈值 -- 相对 a1 (W=16 时 0.9985@0.118/0.9958@0.120)
        是否显著更早/更陡;
    S2  若 G_up 保持 ~1.000 到 W~up gap (2.16@0.120) => 过滤窗连磁性无序
        都免疫 (最强结论);
    S3  若早崩 (W << up gap) => "远离磁性杂质"设计规则
        (sigma_x ~ 磁掺杂/极化杂质, 破坏手性保护).

口径 (与 B3b a1 全同, 仅 DM 与系统升维):
  * 无序只进散射区 onsite += W v SX4, v ~ U(-0.5,0.5); leads 干净.
  * 系统 = 4x4: 轨道序 [upE,upH,dnE,dnH] = (spin s) x (抽象基 a),
    块对角 hoppings (up/dn 的 Floquet 参数不同) + sigma_x(x)I2 onsite.
  * E = auto = -D A0^2. 统计 NC realizations, 报 <G>±std (G=总透射,
    因 dn 出射无模 => 即 up 通道 G_up; 出射极化 P_out=+1).
  * seed 空间隔离 MODE_ID=9 (与 B2/B3a/B3b 的 5/6/7/8 不撞).

运行 (kwant2 终端, 本目录; env 全有默认):
  冒烟 (1 点 x 2 W x NC=4, ~1 min):  python hex_rib_disorder_sx.py
  正式 (推荐): A0=0.120 x W=0,1,2,4,8,16 x NC=12, ~10-20 min 视核数
      set HXS_A0S=0.120& set HXS_W=0,1,2,4,8,16& set HXS_NC=12& set HXS_TAG=sx& python hex_rib_disorder_sx.py
  加靠左缘脆点 (gap 依赖对照):  set HXS_A0S=0.118,0.120&
  env: HXS_A0S=0.120  HXS_NY=160  HXS_LX=60  HXS_W=0,1,2,4,8,16
       HXS_NC=12  HXS_NP=(auto)  HXS_TAG=sx
输出: data/hex_rib_disorder_{TAG}.json, results[A0][W] = 统计块 + 全 conf

*** 铁律: 跑的过程绝对禁止 Ctrl+C (mp.Pool 只补 worker 不补任务).
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
import hex_ribbon_transport as hrt     # TILT/J0/几何 (单一来源)

# ------------------------------ 参数 (env 覆盖) -------------------------------
A0S = [float(x) for x in os.environ.get(
    "HXS_A0S", "0.120").split(",")]
NY = int(os.environ.get("HXS_NY", "160"))
LX = int(os.environ.get("HXS_LX", "60"))
W_LIST = [float(x) for x in os.environ.get(
    "HXS_W", "0,1,2,4,8,16").split(",")]
N_CONF = int(os.environ.get("HXS_NC", "12"))
TAG = os.environ.get("HXS_TAG", "sx")
NPROC = int(os.environ.get("HXS_NP", "0")) or min(os.cpu_count() or 1, 8)

A, B, D, M0, HW = hkb.A, hkb.B, hkb.D, hkb.M0, hkb.HW
SQ3 = hkb.SQ3
TILT = hrt.TILT
J0 = hrt.J0
MODE_ID = 9                                  # seed 空间与 a1(7)/mz(8) 隔离
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "hex_rib_disorder_%s.json" % TAG)

Z2 = np.zeros((2, 2), dtype=complex)
# sigma_x (自旋) (x) I2 (抽象基): 行/列序 (upE,upH,dnE,dnH)
SX4 = np.array([[0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0]], dtype=complex)


def log(msg):
    print(msg, flush=True)


def seed_of(A0v, Wd, c):
    """确定性 seed: (模式, A0, 无序强度, 配置) 唯一."""
    a = int(round(A0v * 1e5))            # 5 位
    w = int(round(Wd * 1e3))             # 5 位 (meV*1000)
    return (MODE_ID * 10**16 + a * 10**8 + w * 10**3 + c)


def Ec(A0v):
    """gap 中心 E_c = -D A0^2 (hrt.energy auto 同口径)."""
    return -D * A0v ** 2


def mats4(A0v):
    """4x4 块对角 (up 块 + dn 块), 各块 = hkb.mats 同构, Floquet 参数各异."""
    ocu, t0u, t1u, t2u = hkb.mats("up", *hkb.floquet("up", A0v), A0v, TILT, J0)
    ocd, t0d, t1d, t2d = hkb.mats("dn", *hkb.floquet("dn", A0v), A0v, TILT, J0)

    def bd(mu, md):
        return np.block([[mu, Z2], [Z2, md]])

    return (bd(ocu, ocd), bd(t0u, t0d), bd(t1u, t1d), bd(t2u, t2d))


def make_system(Ny, Lx, A0v, vmap, Wd):
    """hex ribbon 二端 4x4 器件 + 散射区 sigma_x 无序 (leads 干净)."""
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=4)
    oc, t0, t1, t2 = mats4(A0v)

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
        return oc + Wd * vmap.get(tuple(int(v) for v in site.tag), 0.0) * SX4

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
    """一个 (A0, W, conf): 总透射 G (dn 出射无模 => = G_up, P_out=+1)."""
    rng = np.random.default_rng(seed_of(A0v, Wd, c))
    vmap = {(x, m): float(rng.uniform(-0.5, 0.5))
            for x in range(LX) for m in range(NY)}
    fsys = make_system(NY, LX, A0v, vmap, Wd)
    E = Ec(A0v)
    sm = kwant.smatrix(fsys, E)
    return float(sm.transmission(1, 0))


def _task(args):
    A0v, Wd, c = args
    return one_conf(A0v, Wd, c)


def agg_rows(cfgs):
    """cfgs: [G_up...] -> 统计块. dn 出射恒 0 => P_out 解析 +1."""
    g = np.array(cfgs)
    return {
        "G_avg": round(float(g.mean()), 6),
        "G_std": round(float(g.std()), 6),
        "p_blk": round(float((g < 0.25).mean()), 4),
        "p_flt": round(float((g > 0.75).mean()), 4),
        "G_conf": [round(float(x), 6) for x in g],
        "note": "dn 出射无传播模 (E_c 钉 dn gap) => G=总透射=G_up, P_out=+1",
    }


def dump(results):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                               "tilt": TILT, "J0": J0, "Ny": NY, "Lx": LX,
                               "A0_list": A0S, "W_list": W_LIST,
                               "N_conf": N_CONF,
                               "mode": "sx (sigma_x x I2, 自旋混合/磁性)",
                               "E_note": "auto = -D*A0^2 (gap center)"},
                   "results": results},
                  f, ensure_ascii=False, indent=1)


def main():
    t0 = time.time()
    log("== hex ribbon 无序存活(sx 磁性对照)  A0=%s  Ny=%d  Lx=%d  "
        "W=%s  NC=%d  NPROC=%d ==" % (A0S, NY, LX, W_LIST, N_CONF, NPROC))
    log("  4x4 sigma_x(x)I2;  E = auto (-D A0^2);  dn 出射无模 => G=G_up")
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
                    if i % 6 == 0 or i == N_CONF:
                        log("    [A0=%g W=%g  c=%3d/%d  本组 %5.1f s  累计 %6.0f s]"
                            % (A0v, Wd, i, N_CONF, time.time() - t1,
                               time.time() - t0))
                agg = agg_rows(cfgs)
                results.setdefault("%g" % A0v, {})["%g" % Wd] = agg
                dump(results)      # 逐 (A0,W) 组落盘, 随时可收割
                log("  [A0=%g W=%g]  <G_up>=%.4f±%.4f  p_blk=%.2f p_flt=%.2f  "
                    "[已落盘]"
                    % (A0v, Wd, agg["G_avg"], agg["G_std"],
                       agg["p_blk"], agg["p_flt"]))
    finally:
        pool.close()
        pool.join()
    log("\n[落盘] %s  总用时 %.1f s" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

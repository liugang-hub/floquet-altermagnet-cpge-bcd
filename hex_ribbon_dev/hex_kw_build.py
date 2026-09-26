# -*- coding: utf-8 -*-
r"""
hex_kw_build.py -- Paper H 方向 B v0.2: hex TB 实空间化 + kwant ribbon 色散验证
================================================================================
目标 (新论文线地基):
  Paper H 的 hex 连续/TB 预言目前全是 bulk 物理. 方向 B 把它器件化:
  三角格点 ribbon (kwant) 上的自旋分辨二端输运. 本文件是地基:
  (i)   把 hex_tb_model 的动量空间 H(k) 严格反傅里叶成三角格点 hopping;
  (ii)  三线对照验证 (全部 PASS 才允许进入输运 v0.2):
         V1 bulk : 自写实空间 H_bulk(k) vs hex_tb_model.tb_d/h2x2 解析 (随机 40 k)
         V2 ribbon: kwant lead Bands 色散 vs 自写准一维 H_rib(kx) 对角化
  推导与 hex_tb_model.py (方向 A, 已标定 PASS) 完全同源, 无新物理参数.

-------------------------------- 实空间化推导 ---------------------------------
几何: 三角格点 a=1, a1=(1,0), a2=(1/2, sqrt3/2). 3 个无向 bond (kwant 每个
      bond 只定义一次, 反向自动 t^\dagger):
        d0 = a1        (tag +1,  0)
        d1 = a2        (tag  0, +1)
        d2 = a2 - a1   (tag -1, +1)
      6 最近邻有向方向 = {+d0,+d1,+d2} U {-d0,-d1,-d2}.

TB 基函数 (hex_tb_model, a=1):
  g0 = 6 - sum_{j=0..5} cos(k.dj)            ~ (3/2) k^2         [A1]
  gd = sum_j Cj cos(k.dj), Cj=cos(2phi_j)
     = {1,-1/2,-1/2,1,-1/2,-1/2}             ~ (3/4)(ky^2-kx^2)   [E2]
  v1 = sin(kx) + cos(sqrt3 ky/2) sin(kx/2)   ~ (3/2) kx          [E1, Mx-odd]
     = sin(k.d0) + (1/2)[sin(k.d1) + sin(k.d5)]
  v2 = sqrt3 cos(kx/2) sin(sqrt3 ky/2)       ~ (3/2) ky          [E1]
     = (sqrt3/2)[sin(k.d1) - sin(k.d5)]
  其中 d5 = 300 deg = -d2  (故 sin(k.d5) = -sin(k.d2), cos(k.d5)=cos(k.d2))
  v1 第二项 1/2 来自积化和差: cos(sqrt3 ky/2) sin(kx/2)
     = (1/2)[sin(kx/2 + sqrt3 ky/2) + sin(kx/2 - sqrt3 ky/2)].
     (v0.1 首批 V1 FAIL 根因之一: 曾误作等权 sin(k.d1)+sin(k.d5).)

每自旋 2x2 (sigma 空间, s=+1 up / -1 dn):
  h0 = -D A0^2 -(2D/3) g0 + (2T/3) v1
  m  = Mt -(2B/3) g0 + s J0 gd
  hx = (2 At/3) v1,   hy = (2 At/3) v2
  H  = h0 s0 + m sz + hx sx + hy sy

按 cos(偶)/sin(奇) 分解到 3 bond (sigma0/sigmaz 对称通道 t 实标量,
反厄米奇键通道 t=(i/2)alpha*sigma, 贡献 -alpha sin(k.d) sigma):

  [勘误 v0.2] v0.1 首批 V1 FAIL (max|dE| ~ 2.2e3 meV) 三处根因, 本版修正:
    (a) v1 的 d1/d5 项带 1/2 权重 (见上), 故 hx/tilt 在 d1,d2 键的
        sigma_x/sigma_0 反厄米系数 = 等权假设的一半;
    (b) -(2D/3) g0 = -4D + (4D/3) sum cos, 常数 -4D 必须进 onsite
        (旧版只抵消了 m 通道的 -4B, sigma_0 通道漏了 -4D, 缺 +2048 meV);
    (c) gd 的 ±d 成对权重: sum_6 Cj cos = 2cos(d0) - cos(d1) - cos(d2),
        故 m_AM 的 sz 键权 = sJ0 (d0) / -sJ0/2 (d1,d2), 而非 sJ0/2 / -sJ0/4.

  cos 部分 (对称, H 中系数 = 2 t cos):
    h0_D: -(2D/3) g0 = -4D + (4D/3) sum_{d0,d1,d2} cos
        -> 常数 -4D 进 onsite;  t = 2D/3   (s0, 3 bonds)
    m_B : -(2B/3) g0 = -4B + (4B/3) sum cos
        -> 常数 -4B 进 onsite;  t = 2B/3   (sz, 3 bonds)
    m_AM: s J0 gd = s J0 {2 cos(k.d0) - cos(k.d1) - cos(k.d2)}
        -> t = +s J0      (sz, d0)
           t = -s J0/2   (sz, d1,d2)
    (cos 偶对称: d5 = -d2 自动并入, 上面已用 sum_{d0,d1,d2})
  sin 部分 (反厄米, 系数 -alpha 对照; v1 的 d1/d5 带 1/2):
    hx = (2At/3) v1 = (2At/3) sin(kd0) + (At/3) sin(kd1) - (At/3) sin(kd2)
        -> alpha = -2At/3 (d0), -At/3 (d1), +At/3 (d2)
        -> t_sx = -i At/3 s_x (d0);  -i At/6 s_x (d1);  +i At/6 s_x (d2)
    hy = (2At/3) v2 = (At sqrt3/3) [sin(kd1) + sin(kd2)]
        -> alpha = -At sqrt3/3 (d1,d2)
        -> t_sy = -i At sqrt3/6 s_y (d1,d2)
    tilt (2T/3)v1 = (2T/3) sin(kd0) + (T/3) sin(kd1) - (T/3) sin(kd2)
        -> t_s0 = -i T/3 s0 (d0);  -i T/6 s0 (d1);  +i T/6 s0 (d2)

  onsite (显式常数; cos 项在 k=0 自动回补 +4D/+4B, 已在此抵消):
    h0: (-D A0^2 - 4D) s0 ;   m: (Mt - 4B) sz

最终 kwant hopping (到 +d_b, kwant: syst[site+d_b, site] = t):
  t0 = (2D/3 - iT/3) s0 + (2B/3 + s J0) sz - (i At/3) sx
  t1 = (2D/3 - iT/6) s0 + (2B/3 - s J0/2) sz - (i At/6) sx - (i At sqrt3/6) sy
  t2 = (2D/3 + iT/6) s0 + (2B/3 - s J0/2) sz + (i At/6) sx - (i At sqrt3/6) sy
  onsite = (-D A0^2 - 4D) s0 + (Mt - 4B) sz

自检: V1 用 hex_tb_model 解析 H(k) 对照 (任何系数错误立即暴露).

--------------------------------- 运行 (kwant2) --------------------------------
  python hex_kw_build.py                                  # 冒烟: both spins, Ny=10, A0=0.06
  set HR_SPIN=up& set HR_NY=16& set HR_A0=0.12& python hex_kw_build.py   # 单自旋 / 开关点
  (env 全有默认值: HR_SPIN=both HR_NY=10 HR_A0=0.06 HR_T=30 HR_J0=160)
输出: data/hex_kw_v01.json  (V1/V2 max diff, 每自旋)
"""
import os
import sys
import time
import json

import numpy as np
import kwant

sys.path.insert(0, r"D:/0-我的论文/paper_h_qgeom")          # hex_model (只读)
sys.path.insert(0, r"C:/Users/Gang/WorkBuddy/2026-08-19-15-22-01/hex_tb")
import hex_model as hm                                      # R_HEX/floquet_params
import hex_tb_model as tb                                   # tb_d/h2x2 (标定 PASS)

# ------------------------------ 参数 (env 覆盖) -------------------------------
SPIN = os.environ.get("HR_SPIN", "both").lower()            # up | dn | both
NY = int(os.environ.get("HR_NY", "10"))
A0 = float(os.environ.get("HR_A0", "0.06"))
TILT = float(os.environ.get("HR_T", "30.0"))                # meV.nm  (单位铁律!)
J0 = float(os.environ.get("HR_J0", "160.0"))                # meV
KPTS = int(os.environ.get("HR_KPTS", "201"))                # 色散 kx 点数
SEED = 20260903

assert SPIN in ("up", "dn", "both"), "HR_SPIN 只支持 up|dn|both"

A, B, D, M0, HW = tb.A_PARAM, tb.B_PARAM, tb.D_PARAM, tb.M0_PARAM, tb.HW_PARAM
SQ3 = np.sqrt(3.0)

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def sgn(spin):
    return 1.0 if spin == "up" else -1.0


def floquet(spin, A0v):
    """Floquet 重整化 At/Mt (Paper H 同源, sp=+1 sigma+)."""
    AtD, MtD = hm.floquet_params(A, B, D, M0, HW, A0v, sp=1.0)
    return AtD[spin], MtD[spin]


def mats(spin, At, Mt, A0v, Tt, J0v):
    """返回 (onsite, t0, t1, t2) -- 推导见 docstring (v0.2 修正: 1/2 权重/AM*2/-4D)."""
    s = sgn(spin)
    t0 = ((2.0 * D / 3.0) - 1j * (Tt / 3.0)) * s0 \
        + (2.0 * B / 3.0 + s * J0v) * sz - 1j * (At / 3.0) * sx
    t1 = ((2.0 * D / 3.0) - 1j * (Tt / 6.0)) * s0 \
        + (2.0 * B / 3.0 - s * J0v / 2.0) * sz \
        - 1j * (At / 6.0) * sx - 1j * (At * SQ3 / 6.0) * sy
    t2 = ((2.0 * D / 3.0) + 1j * (Tt / 6.0)) * s0 \
        + (2.0 * B / 3.0 - s * J0v / 2.0) * sz \
        + 1j * (At / 6.0) * sx - 1j * (At * SQ3 / 6.0) * sy
    oc = (-D * A0v ** 2 - 4.0 * D) * s0 + (Mt - 4.0 * B) * sz
    return oc, t0, t1, t2


def H_bulk(kx, ky, spin, At, Mt, A0v, Tt, J0v):
    """自写实空间 bulk H(k) = sum_bond [t e^{ik.d} + t^dag e^{-ik.d}] + onsite.

    支持标量或数组 kx/ky: 数组时返回 (N,...,2,2), 标量返回 (2,2).
    (hex_tb_model 基函数只吃数组, verify_bulk 以 40 点数组调用; H_rib 标量路径不受影响)
    """
    oc, t0, t1, t2 = mats(spin, At, Mt, A0v, Tt, J0v)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    H = np.empty(kx.shape + (2, 2), dtype=complex)
    H[...] = oc
    for t, d in ((t0, (1.0, 0.0)),
                 (t1, (0.5, SQ3 / 2.0)),
                 (t2, (-0.5, SQ3 / 2.0))):
        kd = kx * d[0] + ky * d[1]
        H += t * np.exp(1j * kd)[..., None, None] \
            + t.conj().T * np.exp(-1j * kd)[..., None, None]
    return H


def H_ref(kx, ky, spin, At, Mt, A0v, Tt, J0v):
    """hex_tb_model 解析 (方向 A 标定 PASS 的口径)."""
    h0, hx, hy, m = tb.tb_d(kx, ky, spin, At, Mt, A0v, Tt, J0v)
    return tb.h2x2(h0, hx, hy, m)


# ------------------------------ V1: bulk 三线 --------------------------------
def verify_bulk(spin, At, Mt):
    """40 随机 k 一次向量化: hex_tb_model 基函数 (g0/gd) 只吃数组 (kx[...,None]),
    逐点标量会 TypeError. 返回 max|dE| (两套 2x2 谱按序差的最大)."""
    rng = np.random.default_rng(SEED)
    K = rng.uniform(-2.0, 2.0, size=(40, 2))
    kx, ky = K[:, 0], K[:, 1]
    Ha = H_bulk(kx, ky, spin, At, Mt, A0, TILT, J0)   # (40,2,2)
    Hb = H_ref(kx, ky, spin, At, Mt, A0, TILT, J0)    # (40,2,2)
    ea = np.sort(np.linalg.eigvalsh(Ha), axis=-1)
    eb = np.sort(np.linalg.eigvalsh(Hb), axis=-1)
    return float(np.abs(ea - eb).max())


# ----------------------- V2: ribbon 色散 (kwant vs 自写) ----------------------
def H_rib(kx, Ny, spin, At, Mt):
    """自写准一维 H(kx): 宽 Ny 行 (每行 2 轨道), 沿 x=a1 无限. 相位约定 +kx."""
    oc, t0, t1, t2 = mats(spin, At, Mt, A0, TILT, J0)
    H = np.zeros((2 * Ny, 2 * Ny), dtype=complex)
    eik = np.exp(1j * kx)
    eikm = np.exp(-1j * kx)
    for m in range(Ny):
        a = 2 * m
        H[a:a + 2, a:a + 2] = oc + t0 * eik + t0.conj().T * eikm
        if m + 1 < Ny:
            b = 2 * (m + 1)
            H[a:a + 2, b:b + 2] = t1 + t2 * eikm
            H[b:b + 2, a:a + 2] = t1.conj().T + t2.conj().T * eik
    return H


def build_lead(Ny, spin, At, Mt):
    """kwant 无限 lead: 对称沿 a1=(1,0), 宽 Ny 行 (m=0..Ny-1), 每 site 2 轨道.

    kwant 铁律: hop 首端必须先存在于 builder. 故先加全部 onsite, 再统一加 hops
    (同轮内先 hop 后 onsite 会 KeyError: 如 t1 的 lat(0,m+1) 要到下轮才定义).
    """
    lat = kwant.lattice.Monatomic([(1.0, 0.0), (0.5, SQ3 / 2.0)], norbs=2)
    oc, t0, t1, t2 = mats(spin, At, Mt, A0, TILT, J0)
    lead = kwant.Builder(kwant.TranslationalSymmetry([1.0, 0.0]))
    for m in range(Ny):
        lead[lat(0, m)] = oc
    for m in range(Ny):
        lead[lat(1, m), lat(0, m)] = t0          # d0 = a1
        if m + 1 < Ny:
            lead[lat(0, m + 1), lat(0, m)] = t1      # d1 = a2
            lead[lat(-1, m + 1), lat(0, m)] = t2     # d2 = a2 - a1
    return lead.finalized()


def verify_ribbon(spin, At, Mt):
    fsys = build_lead(NY, spin, At, Mt)
    bands = kwant.physics.Bands(fsys)
    kxs = np.linspace(-np.pi, np.pi, KPTS, endpoint=False)
    d_p, d_m = 0.0, 0.0
    for kx in kxs:
        Ek = np.sort(np.asarray(bands(kx)))
        Es_p = np.sort(np.linalg.eigvalsh(H_rib(kx, NY, spin, At, Mt)))
        Es_m = np.sort(np.linalg.eigvalsh(H_rib(-kx, NY, spin, At, Mt)))
        d_p = max(d_p, np.abs(Ek - Es_p).max())
        d_m = max(d_m, np.abs(Ek - Es_m).max())
    return float(d_p), float(d_m)


# ---------------------------------- main -------------------------------------
def main():
    t0 = time.time()
    spins = ["up", "dn"] if SPIN == "both" else [SPIN]
    print("== hex kwant ribbon build v0.2  spin=%s  Ny=%d  A0=%g  Tilt=%g  J0=%g =="
          % (SPIN, NY, A0, TILT, J0))
    res = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "A0": A0,
                      "tilt": TILT, "J0": J0, "Ny": NY, "Kpts": KPTS,
                      "lattice": "triangular a=1, ribbon along a1, 2 orbs/site"},
           "spins": {}}
    ok_all = True
    for spin in spins:
        At, Mt = floquet(spin, A0)
        dv1 = verify_bulk(spin, At, Mt)
        dv2p, dv2m = verify_ribbon(spin, At, Mt)
        pass1 = dv1 < 1e-7
        pass2 = min(dv2p, dv2m) < 1e-6
        ok_all = ok_all and bool(pass1) and bool(pass2)
        res["spins"][spin] = {"At": At, "Mt": Mt,
                              "V1_bulk_maxde": float(dv1),
                              "V2_bands_maxde_pluskx": float(dv2p),
                              "V2_bands_maxde_minuskx": float(dv2m),
                              "V1_PASS": bool(pass1), "V2_PASS": bool(pass2)}
        print("  spin=%s  At=%.6f Mt=%.4f" % (spin, At, Mt))
        print("    V1 bulk 实空间 vs 解析  max|dE| = %.3e  -> %s"
              % (dv1, "PASS" if pass1 else "FAIL"))
        print("    V2 ribbon kwant vs 自写(+kx) = %.3e  (-kx) = %.3e  -> %s"
              % (dv2p, dv2m, "PASS" if pass2 else "FAIL 见相位说明"))
    res["ALL_PASS"] = bool(ok_all)
    print("=> %s  (%.1f s)" % ("ALL PASS" if ok_all else "有 FAIL", time.time() - t0))
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "hex_kw_v01.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("[ok] data\\hex_kw_v01.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

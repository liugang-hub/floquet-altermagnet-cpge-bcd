# -*- coding: utf-8 -*-
r"""
hex_bcd_temp.py -- 宁波基金补算 P2: BCD/NHE 全温区曲线 + 散射展宽并存
================================================================================
物理 (补报告管线(4)明文缺口: "中间温区曲线与无序、温度并存的情形尚未建立"):
  已有温度证据只有两个端点口径 (T=0 与 T=30 meV 热展宽, hex_bcd_surface
  v1/d8/t30), 中间温区曲线缺失; 且温度展宽与 disorder 展宽从未并存评估.
  口径升级: 占据函数用"Lorentzian 展宽费米函数" (Buttiker 虚导线/寿命展宽
  的标准闭式):
      f~(E) = 1/2 - (1/pi) * Im[ psi( 1/2 + (gamma + i(E-EF)) / (2*pi*kT) ) ]
    (psi = digamma; T=0,gamma=0 -> 台阶; T=0,gamma>0 -> 1/2-arctan((E-EF)/gamma)/pi
     即 Lorentzian 卷积; gamma=0,T>0 -> 精确费米函数; 三极限全部自检)
  => gamma 解释为散射寿命展宽 (与器件线 hex_rib_disorder_dp 的 gamma 同名
    同单位, 器件端 gamma 衰减律 alpha 与体响应端的 gamma 展宽构成同一物理
    的两侧 => 报告"无序、温度并存"缺口一次闭合).

口径 (与 hex_bcd_surface.py d8 档逐条对齐, 只把温度从常数变为扫描轴):
  * 工作点固定 (A0=0.06, hw=150, tilt=30), delta=8 (深入自旋劈裂窗口,
    up/dn 双费米面 => 自旋 BCD 通道全开, d8 主口径);
  * EF = top_max(ref) - delta 固定绝对能量 (与 d8 的 grid 行完全同口径 =>
    (T=0, gamma=0) 行可与 hex_bcd_surface_d8.json grid(A0=0.06,hw=150) 行
    逐位对账, NK 相同时须逐位一致);
  * tilt 扫描轴砍掉 (t30 已有), 温度 T 成为扫描轴: 0..40 meV;
  * gamma 轴: 0,1,2,4 meV (器件端 gamma=4 已钉死端点, 体端同范围对照);
  * 只含占据态 (valence band); 拓扑恒等式自检 (全占带 D=0) 照抄.

运行 (kwant2 终端, 本目录; 纯 numpy/scipy 无 kwant):
  冒烟 (NK=40, ~秒级):  set PH_T_NK=40& set PH_T_TAG=tsmoke& python hex_bcd_temp.py
  正式 (NK=400, 与 d8 同网格可对账, ~1 min):
  set PH_T_NK=400& set PH_T_TAG=temp& python hex_bcd_temp.py
env: PH_T_NK / PH_T_J0 / PH_T_A0 / PH_T_HW / PH_T_TILT / PH_T_DELTA /
     PH_T_TILTS->无 / PH_T_TEMPS (逗号列表, meV) / PH_T_GAMMAS (meV) /
     PH_T_TAG

输出:
  data/hex_bcd_temp_{TAG}.json  (params / topology_check / ef_ref / rows[])
  figs/hex_bcd_temp_{TAG}.png   (2 面板: Dx_tot, Dx_spin vs T, 各 gamma 一条线)
"""
import os
import json
import time

import numpy as np
from scipy.special import digamma as psi

import hex_model as hm

# ------------------------------ 参数 -----------------------------------------
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
SP = 1.0
J0 = float(os.environ.get("PH_T_J0", "160.0"))
NK = int(os.environ.get("PH_T_NK", "120"))
TAG = os.environ.get("PH_T_TAG", "temp")

A0_REF = float(os.environ.get("PH_T_A0", "0.06"))
HW_REF = float(os.environ.get("PH_T_HW", "150.0"))
TILT_REF = float(os.environ.get("PH_T_TILT", "30.0"))
DELTA = float(os.environ.get("PH_T_DELTA", "8.0"))

TEMPS = [float(x) for x in os.environ.get(
    "PH_T_TEMPS", "0,5,10,15,20,25,30,40").split(",")]
GAMMAS = [float(x) for x in os.environ.get(
    "PH_T_GAMMAS", "0,1,2,4").split(",")]

assert min(TEMPS) >= 0.0 and min(GAMMAS) >= 0.0
assert HW_REF >= 50.0, "hw < 50 meV 高频近似失效, 拒绝执行"

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hex_bcd_temp_%s.json" % TAG)
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figs", "hex_bcd_temp_%s.png" % TAG)


def occ_broad(E, EF, T, gam):
    """Lorentzian 展宽费米函数 (闭式, 三极限自检):
    T=0,g=0: 台阶; T=0,g>0: 1/2 - arctan((E-EF)/g)/pi; g=0,T>0: 精确费米."""
    E = np.asarray(E, dtype=float)
    if T <= 1e-9 and gam <= 1e-9:
        return (E < EF).astype(float)
    if T <= 1e-9:
        return 0.5 - np.arctan((E - EF) / gam) / np.pi
    z = 0.5 + (gam + 1j * (E - EF)) / (2.0 * np.pi * T)
    return 0.5 - np.imag(psi(z)) / np.pi


def self_check_occ():
    """三重数值自检: (a) g=0 vs 精确费米; (b) T=0 vs 数值 Lorentzian 卷积;
    (c) T>0,g>0 vs 数值卷积 (Fermi x Lorentzian). 任何一项超差直接退出.
    卷积口径: f~(E) = int L(s) f(E-s) ds, s 在 grid 上."""
    # (a) gamma=0 -> 精确费米函数
    EE = np.linspace(-80.0, 80.0, 41)
    fa = occ_broad(EE, 0.0, 30.0, 0.0)
    fe = 1.0 / (1.0 + np.exp(EE / 30.0))
    da = float(np.max(np.abs(fa - fe)))
    assert da < 1e-10, "occ 自检 (a) 失败: %.2e" % da
    # (b) T=0: 自适应积分 step x Lorentzian; f0(E-s)=1 当 s>E
    g = 5.0
    lor_f = lambda s: g / np.pi / (s ** 2 + g ** 2)
    from scipy.integrate import quad
    for E0 in (-12.0, -3.0, 0.0, 4.0, 15.0):
        num, _ = quad(lor_f, E0, np.inf)
        ana = float(occ_broad(np.array([E0]), 0.0, 0.0, g)[0])
        db = abs(num - ana)
        assert db < 1e-8, "occ 自检 (b) 失败 @E=%g: %.2e" % (E0, db)
    # (c) T=30, g=5: 自适应积分 Fermi(T) x Lorentzian
    T = 30.0
    for E0 in (-20.0, -5.0, 0.0, 6.0, 25.0):
        def integrand(s, E0=E0):
            with np.errstate(over="ignore"):     # exp 大参数溢出=0, 正常
                return lor_f(s) / (1.0 + np.exp((E0 - s) / T))
        num, _ = quad(integrand, -np.inf, np.inf)
        ana = float(occ_broad(np.array([E0]), 0.0, T, g)[0])
        dc = abs(num - ana)
        assert dc < 1e-8, "occ 自检 (c) 失败 @E=%g: %.2e" % (E0, dc)
    print("[occ 自检] (a) 费米极限 %.1e  (b) T=0 积分 <1e-8  (c) 双展宽积分 "
          "<1e-8  全部 PASS" % da, flush=True)


def fermi_step(E, EF, T):
    """同 hex_bcd_surface 的占据 (对照用)."""
    if T <= 1e-9:
        return (E < EF).astype(float)
    return 1.0 / (1.0 + np.exp((E - EF) / T))


def band_top(kx, ky, mask, spin, At, Mt, A0, T):
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, T, J0)
    return float(Ev[mask].max())


def bcd_vec(kx, ky, mask, d2k, spin, At, Mt, A0, TILT_H, T_meV, gam, EF):
    """BCD D_a = sum f~(E_v) d_a Omega * d2k, f~ = 展宽占据 (向量化)."""
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    _, dOx, dOy = hm.omega_full(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    f = occ_broad(Ev, EF, T_meV, gam)
    occ = float((f * mask).sum()) / float(mask.sum())
    Dx = float(np.sum(f * dOx * mask) * d2k)
    Dy = float(np.sum(f * dOy * mask) * d2k)
    return Dx, Dy, occ


def floquet(A0, HW):
    return hm.floquet_params(A, B, D, M0, HW, A0, sp=SP)


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    self_check_occ()

    kx, ky, mask, d2k = hm.hex_mesh(NK)
    print("== hex BCD/NHE 全温区 x 展宽   NK=%d   J0=%.0f  (A0=%.2f, hw=%.0f, "
          "tilt=%.0f, delta=%.1f)  T=%s  gamma=%s =="
          % (NK, J0, A0_REF, HW_REF, TILT_REF, DELTA, TEMPS, GAMMAS),
          flush=True)

    # ---- 拓扑恒等式自检 (全占带 D=0) ----
    At0, Mt0 = floquet(A0_REF, HW_REF)
    _, dOx0, dOy0 = hm.omega_full(kx, ky, "up", At0, Mt0, A0_REF, TILT_REF, J0)
    chk = {"Dx_all_occupied": float(np.sum(dOx0 * mask) * d2k),
           "Dy_all_occupied": float(np.sum(dOy0 * mask) * d2k)}
    print("[拓扑恒等式] 全占带 Dx=%.2e Dy=%.2e (须 ~0)"
          % (chk["Dx_all_occupied"], chk["Dy_all_occupied"]), flush=True)

    # ---- EF 锚定: 参考点 valence 带顶 - delta (与 d8 grid 行同口径) ----
    top_ref = max(band_top(kx, ky, mask, s, At0, Mt0, A0_REF, TILT_REF)
                  for s in ("up", "dn"))
    EF = top_ref - DELTA
    print("参考带顶 top_max=%.4f meV -> EF=%.4f meV (固定, 全温区共享)"
          % (top_ref, EF), flush=True)

    # ---- (T, gamma) 网格 ----
    rows = []
    for T in TEMPS:
        for gam in GAMMAS:
            spin_rows = {}
            for s in ("up", "dn"):
                Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, s, At0, Mt0,
                                      A0_REF, TILT_REF, T, gam, EF)
                spin_rows[s] = {"Dx": Dx, "Dy": Dy, "occ": occ}
            r = {"T": T, "gamma": gam, "EF": EF,
                 "Dx_up": spin_rows["up"]["Dx"],
                 "Dx_dn": spin_rows["dn"]["Dx"],
                 "Dy_up": spin_rows["up"]["Dy"],
                 "Dy_dn": spin_rows["dn"]["Dy"],
                 "occ_up": spin_rows["up"]["occ"],
                 "occ_dn": spin_rows["dn"]["occ"]}
            r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
            r["Dx_spin"] = r["Dx_up"] - r["Dx_dn"]
            r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
            rows.append(r)
            print("  T=%5.1f g=%4.1f:  Dx(tot,spin)=(%+.3e,%+.3e)  "
                  "Dy_tot=%+.2e  occ=(%.4f,%.4f)"
                  % (T, gam, r["Dx_tot"], r["Dx_spin"], r["Dy_tot"],
                     r["occ_up"], r["occ_dn"]), flush=True)

    # ---- 电荷保留 / 自旋压制比 (gamma=0 列, 报告叙事直接可用) ----
    ref = next(r for r in rows if r["T"] == 0.0 and r["gamma"] == 0.0)
    ratio = []
    for T in TEMPS:
        r0 = next(r for r in rows if r["T"] == T and r["gamma"] == 0.0)
        ratio.append({"T": T,
                      "charge_retain": (r0["Dx_tot"] / ref["Dx_tot"]
                                        if abs(ref["Dx_tot"]) > 1e-30
                                        else None),
                      "spin_retain": (r0["Dx_spin"] / ref["Dx_spin"]
                                      if abs(ref["Dx_spin"]) > 1e-30
                                      else None)})
    print("\n[gamma=0 温度指纹] %s" % ratio, flush=True)

    # ---- json ----
    out = {"params": {"A": A, "B": B, "D": D, "M0": M0, "SP": SP, "J0": J0,
                      "NK": NK, "A0_ref": A0_REF, "hw_ref": HW_REF,
                      "tilt_ref": TILT_REF, "delta": DELTA,
                      "T_list": TEMPS, "gamma_list": GAMMAS,
                      "lattice": "triangular, hexagonal 1st BZ",
                      "occupancy": "Lorentzian-broadened Fermi (digamma "
                                   "closed form); gamma = lifetime/scattering "
                                   "broadening, same unit as device-line "
                                   "dephasing gamma",
                      "model": "continuous k.p + hex BZ (v2.1); "
                               "fd=(3/8)(ky^2-kx^2); EF fixed at "
                               "ref-top - delta (d8 caliber)"},
           "topology_check": chk, "ef_ref": EF, "top_ref": top_ref,
           "rows": rows, "temp_ratio_g0": ratio}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n[落盘] %s" % OUT_JSON, flush=True)

    # ---- 图 (2 面板, 英文标题避 CJK 缺字形) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    colors = {0.0: "k", 1.0: "tab:blue", 2.0: "tab:red", 4.0: "tab:green"}
    for ax, key, lab in ((axes[0], "Dx_tot", r"$D_x^{\rm tot}$ (charge NHE)"),
                         (axes[1], "Dx_spin", r"$D_x^{\rm spin}$ (spin NHE)")):
        for gam in GAMMAS:
            sub = [r for r in rows if r["gamma"] == gam]
            ax.plot([r["T"] for r in sub], [r[key] for r in sub], "o-",
                    lw=1.4, ms=4,
                    color=colors.get(gam, None),
                    label=r"$\gamma=%.0f$ meV" % gam)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(r"$k_BT$ (meV)")
        ax.set_ylabel(lab)
        ax.set_title(r"$A_0=0.06$, $\hbar\omega=150$, tilt$=30$, "
                     r"$\delta=8$ meV")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.grid(alpha=0.3)
    fig.suptitle("BCD vs temperature and scattering broadening "
                 r"($J_0=%.0f$ meV, $N_K=%d$)" % (J0, NK))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print("[落盘] %s  总用时 %.1f s" % (OUT_PNG, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

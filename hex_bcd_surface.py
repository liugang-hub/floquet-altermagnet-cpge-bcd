# -*- coding: utf-8 -*-
r"""
hex_bcd_surface.py -- 宁波基金补算 ②: BCD/NHE 响应面细化 (tilt 扫描 + (A0,hw) 网格)
====================================================================================
物理 (hex_model v2.1 六角主口径, 与 hex_bcd_nhe.py 同源同口径):
  六角 altermagnet (MnTe 取向 fd=(3/8)(ky^2-kx^2)) 下 BCD:
    * tilt=0: fd 双镜面偶 + Omega 对 kx,ky 分别偶 => 残留 C2v => D_x=D_y=0 严格;
    * tilt T*kx 只破 kx 镜面 (My 保留) => D_y=0 精确, D_x 打开 (NHE J_y~D_x E_x^2);
    * J0 项使 D_x^up != D_x^dn => 自旋 BCD 通道 D_x^s = D_x^up - D_x^dn;
    * Floquet (A0, hw): 手性劈裂 Delta_Mt = 2 A0^2 A^2 / hw ~ 1/hw 连续调质量项
      => (A0,hw) 网格 = 响应面; D_x 对 Delta_Mt 的单调性直接印证报告里的
      劈裂对称性公式链 (质量项重正化 -> 自旋分辨费米面 -> BCD 响应).

口径 (逐条对齐 hex_bcd_nhe.py):
  * BCD D_a = sum f(E_v,EF,T) d_a Omega * d2k;  f = Fermi 台阶/函数;
    T 默认 0 (纯费米面口径, 与 PaperH BCD 主结论/报告公式链一致);
    设 PH_SURF_TEMP=30 可与 hex_bcd_nhe 室温行 (a0_scan, T=30) 对照;
  * EF 口径分两层 (都记录在 json):
    - tilt 扫描: 每行 EF = top_max(T) - delta (相对带顶深度钉住; T*kx 移动带顶,
      固定 EF 会混入费米深度漂移 => D(T) 标度须同深度口径);
    - (A0,hw) 网格: EF = top_max(ref) - delta 固定绝对能量 (化学势固定,
      展示光调响应面; 默认 delta=3 meV 时 dn 近似全占 => 想同时开 up/dn
      费米面做自旋 BCD 通道, 设 PH_SURF_DELTA=8 深入自旋劈裂窗口 3.76 meV).
  * 只含占据态 (valence band E_v = h0 - |d|); 拓扑恒等式自检 (全占带 D=0) 照抄.
  * spin 独立求解, 自旋分辨输出.

网格 (默认, 用户可 env 覆盖):
  tilt 扫描: T = 0,10,20,30,40,50 meV  @ (A0=0.06, hw=150), T=0 行自检 D_x ~ 0;
  (A0,hw) 网格: A0 = 0..0.12 (7 点) x hw = 100..250 meV (7 点) @ tilt=30.
  NK 纪律: 冒烟 PH_NK=120, 正式 PH_NK=400 (BCD 收敛网格, 同 hex_bcd_nhe_official).

运行 (kwant2/plain python 均可, 本目录; 纯 numpy 无 kwant):
  冒烟:  set PH_NK=40& python hex_bcd_surface.py
  正式 (delta=3, 与 hex_bcd_nhe_official 口径对账):
  set PH_NK=400& set PH_SURF_TAG=v1& python hex_bcd_surface.py
  对照 (delta=8, 深入自旋劈裂窗口, up/dn 双费米面 => 自旋 BCD 通道全开):
  set PH_NK=400& set PH_SURF_TAG=d8& set PH_SURF_DELTA=8& python hex_bcd_surface.py
  室温对照 (与 a0_scan T=30 行拼接):
  set PH_NK=400& set PH_SURF_TAG=t30& set PH_SURF_TEMP=30& python hex_bcd_surface.py
env: PH_NK / PH_J0 / PH_SURF_TILTS (逗号列表) / PH_SURF_A0S / PH_SURF_HWS /
     PH_SURF_TILT (网格固定 tilt, 默认 30) / PH_SURF_A0_REF / PH_SURF_HW_REF /
     PH_SURF_DELTA (EF 深度, 默认 3.0) / PH_SURF_TEMP (默认 30.0) /
     PH_SURF_TAG (默认 v1)

输出:
  data/hex_bcd_surface_{TAG}.json  (params / ef_ref / tilt_scan[] / grid[])
  figs/hex_bcd_surface_{TAG}.png   (6 面板: tilt 扫描; (A0,hw) 热图 Dx_tot/Dx_spin;
                                     Dx_tot vs A0 切面; Dx_tot vs Delta_Mt 散点;
                                     Mt_up/dn vs hw 质量项重正化路径)
"""
import os, json, time
import numpy as np

import hex_model as hm

# ------------------------------ 参数 -----------------------------------------
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
HW_PARAM = hm.HW_PARAM
SP = 1.0
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "120"))
TAG = os.environ.get("PH_SURF_TAG", "v1")

# 参考工作点 (EF 锚定) 与热占据
A0_REF = float(os.environ.get("PH_SURF_A0_REF", "0.06"))
HW_REF = float(os.environ.get("PH_SURF_HW_REF", "150.0"))
TILT_REF = float(os.environ.get("PH_SURF_TILT", "30.0"))
DELTA = float(os.environ.get("PH_SURF_DELTA", "3.0"))
TEMP = float(os.environ.get("PH_SURF_TEMP", "0.0"))    # k_B T (meV); 0 = T=0 台阶

# 扫描网格
TILTS = [float(x) for x in os.environ.get(
    "PH_SURF_TILTS", "0,10,20,30,40,50").split(",")]
A0S = [float(x) for x in os.environ.get(
    "PH_SURF_A0S", "0,0.02,0.04,0.06,0.08,0.10,0.12").split(",")]
HWS = [float(x) for x in os.environ.get(
    "PH_SURF_HWS", "100,125,150,175,200,225,250").split(",")]
assert min(HWS) >= 50.0, "hw < 50 meV 高频近似失效, 拒绝执行"

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hex_bcd_surface_%s.json" % TAG)
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figs", "hex_bcd_surface_%s.png" % TAG)


def fermi_step(E, EF, T):
    """Occupancy: step at T=0, Fermi function at T>0 (同 hex_bcd_nhe)."""
    if T <= 1e-9:
        return (E < EF).astype(float)
    return 1.0 / (1.0 + np.exp((E - EF) / T))


def band_top(kx, ky, mask, spin, At, Mt, A0, T):
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, T, J0)
    return float(Ev[mask].max())


def bcd_vec(kx, ky, mask, d2k, spin, At, Mt, A0, TILT_H, T_meV, EF):
    """BCD D_a = sum f(E_v) d_a Omega * d2k (向量化, 同 hex_bcd_nhe)."""
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    _, dOx, dOy = hm.omega_full(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    f = fermi_step(Ev, EF, T_meV)
    occ = float((f * mask).sum()) / float(mask.sum())
    Dx = float(np.sum(f * dOx * mask) * d2k)
    Dy = float(np.sum(f * dOy * mask) * d2k)
    return Dx, Dy, occ


def floquet(A0, HW):
    return hm.floquet_params(A, B, D, M0, HW, A0, sp=SP)


def row_of(spin_rows, base):
    """spin_rows: {up: {Dx,Dy,occ}, dn: ...} -> 汇总 row (in-place base)."""
    r = dict(base)
    for s in ("up", "dn"):
        r["Dx_%s" % s] = spin_rows[s]["Dx"]
        r["Dy_%s" % s] = spin_rows[s]["Dy"]
        r["occ_%s" % s] = spin_rows[s]["occ"]
    r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
    r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
    r["Dx_spin"] = r["Dx_up"] - r["Dx_dn"]
    return r


def main():
    t0 = time.time()
    os.makedirs(os.path.join(os.path.dirname(OUT_JSON)), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(OUT_PNG)), exist_ok=True)

    kx, ky, mask, d2k = hm.hex_mesh(NK)
    print("== hex BCD/NHE 响应面   NK=%d   J0=%.0f  (A0_ref=%.2f, hw_ref=%.0f, "
          "tilt_ref=%.0f, delta=%.1f, T=%.0f meV) =="
          % (NK, J0, A0_REF, HW_REF, TILT_REF, DELTA, TEMP), flush=True)

    # ---- 拓扑恒等式自检 (全占带 D=0) ----
    At0, Mt0 = floquet(A0_REF, HW_REF)
    _, dOx0, dOy0 = hm.omega_full(kx, ky, "up", At0, Mt0, A0_REF, TILT_REF, J0)
    chk = {"Dx_all_occupied": float(np.sum(dOx0 * mask) * d2k),
           "Dy_all_occupied": float(np.sum(dOy0 * mask) * d2k)}
    print("[拓扑恒等式] 全占带 Dx=%.2e Dy=%.2e (须 ~0)"
          % (chk["Dx_all_occupied"], chk["Dy_all_occupied"]), flush=True)

    # ---- EF 锚定: 参考点 valence 带顶 - delta ----
    top_ref = max(band_top(kx, ky, mask, s, At0, Mt0, A0_REF, TILT_REF)
                  for s in ("up", "dn"))
    EF = top_ref - DELTA
    print("参考带顶 top_max=%.4f meV -> EF=%.4f meV (固定, 全网格共享)"
          % (top_ref, EF), flush=True)

    # ================= (a) tilt 扫描 @ (A0_ref, hw_ref) =================
    # 口径: tilt 的 T*kx 线性项会移动 valence 带顶 => 每行钉 EF = top_max(T)-delta
    # (相对深度固定, D(T) 标度才干净; 同 hex_bcd_nhe ef_scan 的 CTRL 行逻辑).
    print("\n== tilt 扫描 (T 破 kx 镜面, 每行 EF=top_max(T)-%.1f; T=0 行 D_x 应 ~0) =="
          % DELTA, flush=True)
    tilt_rows = []
    for T in TILTS:
        topT = max(band_top(kx, ky, mask, s, At0, Mt0, A0_REF, T)
                   for s in ("up", "dn"))
        EF_T = topT - DELTA
        spin_rows = {}
        for s in ("up", "dn"):
            Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, s, At0, Mt0, A0_REF,
                                  T, TEMP, EF_T)
            spin_rows[s] = {"Dx": Dx, "Dy": Dy, "occ": occ}
        r = row_of(spin_rows, {"T": T, "EF": EF_T, "top_max": topT})
        tilt_rows.append(r)
        print("  T=%5.1f: top=%+6.3f EF=%+6.3f  "
              "Dx(up,dn,tot,spin)=(%+.3e,%+.3e,%+.3e,%+.3e)  "
              "Dy_tot=%+.2e  occ=(%.3f,%.3f)"
              % (T, topT, EF_T, r["Dx_up"], r["Dx_dn"], r["Dx_tot"],
                 r["Dx_spin"], r["Dy_tot"], r["occ_up"], r["occ_dn"]),
              flush=True)

    # ================= (b) (A0, hw) 网格 @ tilt 固定 =================
    print("\n== (A0,hw) 响应面网格  A0=%s  hw=%s  tilt=%.0f =="
          % (A0S, HWS, TILT_REF), flush=True)
    grid = []
    for A0v in A0S:
        for HW in HWS:
            At, Mt = floquet(A0v, HW)
            spin_rows = {}
            for s in ("up", "dn"):
                Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, s, At, Mt, A0v,
                                      TILT_REF, TEMP, EF)
                spin_rows[s] = {"Dx": Dx, "Dy": Dy, "occ": occ}
            r = row_of(spin_rows, {"A0": A0v, "hw": HW,
                                   "Mt_up": Mt["up"], "Mt_dn": Mt["dn"],
                                   "At_up": At["up"], "At_dn": At["dn"]})
            r["dMt"] = Mt["dn"] - Mt["up"]      # 手性质量劈裂 (meV)
            grid.append(r)
            print("  A0=%.2f hw=%5.0f: dMt=%+6.3f  Dx(tot,spin)=(%+.3e,%+.3e)  "
                  "Dy_tot=%+.2e  occ=(%.3f,%.3f)"
                  % (A0v, HW, r["dMt"], r["Dx_tot"], r["Dx_spin"],
                     r["Dy_tot"], r["occ_up"], r["occ_dn"]), flush=True)

    # ================= tilt 标度拟合 (报告公式链用) =================
    fit = {}
    for key in ("Dx_tot", "Dx_spin", "Dx_up", "Dx_dn"):
        xs = np.array([r["T"] for r in tilt_rows if r["T"] >= 10.0])
        ys = np.array([r[key] for r in tilt_rows if r["T"] >= 10.0])
        if len(xs) >= 2:
            c = np.polyfit(xs, ys, 1)
            yhat = np.polyval(c, xs)
            ss = 1.0 - float(np.sum((ys - yhat) ** 2)
                             / max(np.sum((ys - ys.mean()) ** 2), 1e-30))
            fit[key] = {"slope": float(c[0]), "intercept": float(c[1]),
                        "R2_linear": round(ss, 5)}
    print("\n[tilt 标度拟合, T>=10 meV 段] %s" % fit, flush=True)

    # ================= json =================
    out = {"params": {"A": A, "B": B, "D": D, "M0": M0, "SP": SP,
                      "J0": J0, "NK": NK, "A0_ref": A0_REF, "hw_ref": HW_REF,
                      "tilt_ref": TILT_REF, "delta": DELTA, "T_meV": TEMP,
                      "tilt_list": TILTS, "A0_list": A0S, "hw_list": HWS,
                      "lattice": "triangular, hexagonal 1st BZ, R = 4pi/3",
                      "model": "continuous k.p + hex BZ (v2.1); "
                               "fd=(3/8)(ky^2-kx^2); tilt T*kx breaks kx mirror; "
                               "EF fixed (chem. pot.) at ref-top - delta"},
           "topology_check": chk, "ef_ref": EF, "top_ref": top_ref,
           "tilt_scan": tilt_rows, "grid": grid, "tilt_fit": fit}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\n[落盘] %s" % OUT_JSON, flush=True)

    # ================= 图 (6 面板) =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.0, 7.6))
    # (a) tilt 扫描
    ax = fig.add_subplot(231)
    ts = [r["T"] for r in tilt_rows]
    ax.plot(ts, [r["Dx_up"] for r in tilt_rows], "ro-", lw=1.4, ms=4,
            label=r"$D_x^{\uparrow}$")
    ax.plot(ts, [r["Dx_dn"] for r in tilt_rows], "bo-", lw=1.4, ms=4,
            label=r"$D_x^{\downarrow}$")
    ax.plot(ts, [r["Dx_tot"] for r in tilt_rows], "k--s", lw=1.2, ms=3,
            label=r"$D_x^{\rm tot}$")
    ax.plot(ts, [r["Dx_spin"] for r in tilt_rows], "g^-", lw=1.4, ms=4,
            label=r"$D_x^{s}$")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"tilt $T$ (meV)")
    ax.set_ylabel(r"$D_x$ (BCD)")
    ax.set_title(r"tilt 扫描 @ $A_0=%.2f,\ \hbar\omega=%.0f$"
                 % (A0_REF, HW_REF))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # (b)(c) (A0,hw) 热图
    Zt = np.zeros((len(A0S), len(HWS)))
    Zs = np.zeros((len(A0S), len(HWS)))
    for r in grid:
        i = A0S.index(r["A0"]); j = HWS.index(r["hw"])
        Zt[i, j] = r["Dx_tot"]; Zs[i, j] = r["Dx_spin"]
    for ax, Z, tt in ((fig.add_subplot(232), Zt, r"$D_x^{\rm tot}$"),
                      (fig.add_subplot(233), Zs, r"$D_x^{s}$")):
        vmax = max(1e-30, np.abs(Z).max())
        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[HWS[0], HWS[-1], A0S[0], A0S[-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel(r"$\hbar\omega$ (meV)")
        ax.set_ylabel(r"$A_0$ (nm$^{-1}$)")
        ax.set_title(tt)
        plt.colorbar(im, ax=ax, shrink=0.85)
    # (d) Dx_tot vs A0 各 hw 切面
    ax = fig.add_subplot(234)
    for HW in HWS:
        sub = [r for r in grid if r["hw"] == HW]
        ax.plot([r["A0"] for r in sub], [r["Dx_tot"] for r in sub],
                "o-", lw=1.1, ms=3, label=r"$\hbar\omega=%.0f$" % HW)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax.set_ylabel(r"$D_x^{\rm tot}$")
    ax.set_title(r"响应面切面, tilt=%.0f meV" % TILT_REF)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    # (e) Dx_tot vs dMt 散点 (色 = hw)
    ax = fig.add_subplot(235)
    sc = ax.scatter([r["dMt"] for r in grid], [r["Dx_tot"] for r in grid],
                    c=[r["hw"] for r in grid], cmap="viridis", s=22)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\Delta M_t=M_t^{\downarrow}-M_t^{\uparrow}$ (meV)")
    ax.set_ylabel(r"$D_x^{\rm tot}$")
    ax.set_title(r"BCD 对手性质量劈裂的响应")
    plt.colorbar(sc, ax=ax, shrink=0.85, label=r"$\hbar\omega$ (meV)")
    ax.grid(alpha=0.3)
    # (f) Mt_up/dn vs hw (质量项重正化路径, A0 色)
    ax = fig.add_subplot(236)
    for A0v in A0S:
        sub = [r for r in grid if r["A0"] == A0v]
        ax.plot([r["hw"] for r in sub], [r["Mt_up"] for r in sub], "^-",
                lw=1.0, ms=3.5, color=plt.cm.viridis(A0v / max(A0S)),
                label=r"$\uparrow,\ A_0=%.2f$" % A0v)
        ax.plot([r["hw"] for r in sub], [r["Mt_dn"] for r in sub], "o-",
                lw=1.0, ms=3, color=plt.cm.viridis(A0v / max(A0S)),
                alpha=0.45, label=r"$\downarrow,\ A_0=%.2f$" % A0v)
    ax.set_xlabel(r"$\hbar\omega$ (meV)")
    ax.set_ylabel(r"$M_t$ (meV)")
    ax.set_title(r"质量项重正化 (劈裂 ~ $A_0^2/\hbar\omega$)")
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(rf"hex BCD/NHE 响应面 ($J_0={J0:.0f}$ meV, "
                 rf"$\delta=E_{{top}}-E_F={DELTA:.1f}$ meV, "
                 rf"$k_BT={TEMP:.0f}$ meV, $N_K={NK}$)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print("[落盘] %s  总用时 %.1f s" % (OUT_PNG, time.time() - t0))


if __name__ == "__main__":
    main()

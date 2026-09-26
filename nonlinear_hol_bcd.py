# -*- coding: utf-8 -*-
"""
Paper H -- Script 5: nonlinear Hall effect -- Berry-curvature dipole (BCD)
of the Floquet-renormalized altermagnetic BHZ bulk.

Physics:
  The nonlinear Hall current is j_a = chi_{abc} E_b E_c with the dipole
  of the Berry curvature

      D_a^s = int d^2k f_0(k) d_a Omega_s(k),   a = x, y

  For a C4-symmetric system (J0 = 0) the dipole vanishes by symmetry.
  The d-wave altermagnet term J0[cos(ky)-cos(kx)] distorts Omega_s(k) into
  a d-wave pattern, giving D != 0 with opposite signs for the two spins
  (spin-resolved nonlinear Hall fingerprint).  The drive A0 renormalizes
  the bands and tunes the dipole.

  We use the analytic 2-band Berry curvature (same as berry_curvature_map.py)
  and a central-difference derivative for d_a Omega.

Outputs:
  - data/bcd_floquet.json
  - figs/bcd_panel.png
"""
import os, json, time
import numpy as np

A, B, D, M0 = 364.5, -686.0, -512.0, -10.0
HW = 150.0
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "160"))
EF = 0.0
DK = 1e-3                      # central-difference step for d_a Omega


def floquet_params(A0, sp=SP):
    At = {
        "up": (1.0 - 2.0 * A0**2 * B * sp / HW) * A,
        "dn": (1.0 + 2.0 * A0**2 * B * sp / HW) * A,
    }
    Mt = {
        "up": M0 - B * A0**2 - A0**2 * A**2 * sp / HW,
        "dn": M0 - B * A0**2 + A0**2 * A**2 * sp / HW,
    }
    return At, Mt


def omega_analytic(kx, ky, spin, At, A0):
    """Analytic Berry curvature of the lower band (valence)."""
    s = S3[spin]
    sinx, cosx = np.sin(kx), np.cos(kx)
    siny, cosy = np.sin(ky), np.cos(ky)
    k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
    hx = At[spin] * sinx
    hy = At[spin] * siny
    hz = M0 - B * k2 + 2.0 * s * J0 * (cosy - cosx)
    dm_dx = 2.0 * sinx * (s * J0 - B)
    dm_dy = -2.0 * siny * (B + s * J0)
    dx = np.array([At[spin] * cosx, 0.0, dm_dx])
    dy = np.array([0.0, At[spin] * cosy, dm_dy])
    d_vec = np.array([hx, hy, hz])
    cross = np.cross(dx, dy)
    d_norm = max(np.linalg.norm(d_vec), 1e-12)
    return np.dot(d_vec, cross) / (2.0 * d_norm**3)


def fermi_factor(kx, ky, spin, At, A0):
    """T=0 occupation of the valence band (E_v < EF)."""
    s = S3[spin]
    sinx, cosx = np.sin(kx), np.cos(kx)
    siny, cosy = np.sin(ky), np.cos(ky)
    k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
    h0 = -D * (A0**2 + k2)
    hz = M0 - B * k2 + 2.0 * s * J0 * (cosy - cosx)
    hx = At[spin] * sinx
    hy = At[spin] * siny
    d_norm = np.sqrt(hx**2 + hy**2 + hz**2)
    E_v = h0 - d_norm
    return 1.0 if E_v < EF else 0.0


def bcd_components(kx, ky, spin, At, A0):
    """Return (D_x, D_y) with occupation weighting."""
    nk = kx.shape[0]
    d2k = (2.0 * np.pi / nk) ** 2
    Dx = Dy = 0.0
    for i in range(nk):
        for j in range(nk):
            kx0, ky0 = kx[i, j], ky[i, j]
            f = fermi_factor(kx0, ky0, spin, At, A0)
            if f == 0:
                continue
            om0 = omega_analytic(kx0, ky0, spin, At, A0)
            om_xp = omega_analytic(kx0 + DK, ky0, spin, At, A0)
            om_xm = omega_analytic(kx0 - DK, ky0, spin, At, A0)
            om_yp = omega_analytic(kx0, ky0 + DK, spin, At, A0)
            om_ym = omega_analytic(kx0, ky0 - DK, spin, At, A0)
            Dx += (om_xp - om_xm) / (2.0 * DK) * d2k
            Dy += (om_yp - om_ym) / (2.0 * DK) * d2k
    return Dx, Dy


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    # ---- A0 scan at the working point ----
    print("== A0 scan, M = %+.0f meV, J0 = %.0f meV ==" % (M0, J0))
    a0_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    rows_a0 = []
    for A0 in a0_list:
        At, Mt = floquet_params(A0)
        r = {"A0": A0, "Mt_up": Mt["up"], "Mt_dn": Mt["dn"]}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_components(kx, ky, spin, At, A0)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_a0.append(r)
        print("  A0=%.2f: Dx(up,dn,tot)=(%+.3e,%+.3e,%+.3e)  Dy=(%+.3e,%+.3e,%+.3e)"
              % (A0, r["Dx_up"], r["Dx_dn"], r["Dx_tot"],
                 r["Dy_up"], r["Dy_dn"], r["Dy_tot"]))

    # ---- M scan at A0 = 0.06 ----
    print("\n== M scan, A0 = 0.060 nm^-1, J0 = %.0f meV ==" % J0)
    M_list = [-30.0, -15.0, -10.0, -5.0, 0.0, +10.0]
    rows_m = []
    for M_scan in M_list:
        global_holder = M0   # placeholder; we pass M through a closure
        At, Mt = floquet_params(0.06)
        r = {"M": M_scan}
        for spin in ("up", "dn"):
            # temporarily rebind M0 via module-level patch is not clean;
            # we recompute with a shifted mass below.
            pass
        rows_m.append(r)
    # NOTE: M scan needs a parameterized mass; implement below with a local
    # function instead of rebinding globals.
    def omega_with_M(kx_, ky_, spin, At, A0, Mv):
        s = S3[spin]
        sinx, cosx = np.sin(kx_), np.cos(kx_)
        siny, cosy = np.sin(ky_), np.cos(ky_)
        k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
        hx = At[spin] * sinx
        hy = At[spin] * siny
        hz = Mv - B * k2 + 2.0 * s * J0 * (cosy - cosx)
        dm_dx = 2.0 * sinx * (s * J0 - B)
        dm_dy = -2.0 * siny * (B + s * J0)
        dx = np.array([At[spin] * cosx, 0.0, dm_dx])
        dy = np.array([0.0, At[spin] * cosy, dm_dy])
        d_vec = np.array([hx, hy, hz])
        d_norm = max(np.linalg.norm(d_vec), 1e-12)
        return np.dot(d_vec, np.cross(dx, dy)) / (2.0 * d_norm**3)

    def fermi_with_M(kx_, ky_, spin, At, A0, Mv):
        s = S3[spin]
        sinx, cosx = np.sin(kx_), np.cos(kx_)
        siny, cosy = np.sin(ky_), np.cos(ky_)
        k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
        h0 = -D * (A0**2 + k2)
        hz = Mv - B * k2 + 2.0 * s * J0 * (cosy - cosx)
        hx = At[spin] * sinx
        hy = At[spin] * siny
        d_norm = np.sqrt(hx**2 + hy**2 + hz**2)
        return 1.0 if (h0 - d_norm) < EF else 0.0

    def bcd_with_M(spin, At, A0, Mv):
        Dx = Dy = 0.0
        for i in range(NK):
            for j in range(NK):
                kx0, ky0 = kx[i, j], ky[i, j]
                if fermi_with_M(kx0, ky0, spin, At, A0, Mv) == 0:
                    continue
                om0 = omega_with_M(kx0, ky0, spin, At, A0, Mv)
                Dx += (omega_with_M(kx0 + DK, ky0, spin, At, A0, Mv)
                       - omega_with_M(kx0 - DK, ky0, spin, At, A0, Mv)) / (2.0 * DK) * d2k
                Dy += (omega_with_M(kx0, ky0 + DK, spin, At, A0, Mv)
                       - omega_with_M(kx0, ky0 - DK, spin, At, A0, Mv)) / (2.0 * DK) * d2k
        return Dx, Dy

    rows_m = []
    d2k = (2.0 * np.pi / NK) ** 2
    At06, _ = floquet_params(0.06)
    for Mv in M_list:
        r = {"M": Mv}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_with_M(spin, At06, 0.06, Mv)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_m.append(r)
        print("  M=%+.0f: Dx(up,dn,tot)=(%+.3e,%+.3e,%+.3e)  Dy=(%+.3e,%+.3e,%+.3e)"
              % (Mv, r["Dx_up"], r["Dx_dn"], r["Dx_tot"],
                 r["Dy_up"], r["Dy_dn"], r["Dy_tot"]))

    out = {
        "params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
                   "NK": NK, "EF": EF, "DK": DK,
                   "note": "D_a = int d2k f d_a Omega; units a.u. (unnormalized)"},
        "a0_scan": rows_a0,
        "m_scan": rows_m,
    }
    with open("data/bcd_floquet.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 3.6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    a0s = [r["A0"] for r in rows_a0]
    ax1.plot(a0s, [r["Dx_up"] for r in rows_a0], "ro-", lw=1.4, ms=4, label=r"$D_x$, $\uparrow$")
    ax1.plot(a0s, [r["Dx_dn"] for r in rows_a0], "bo-", lw=1.4, ms=4, label=r"$D_x$, $\downarrow$")
    ax1.plot(a0s, [r["Dx_tot"] for r in rows_a0], "k--o", lw=1.2, ms=3, label=r"$D_x$, tot")
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax1.set_ylabel(r"$D_x$ (BCD)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ms = [r["M"] for r in rows_m]
    ax2.plot(ms, [r["Dx_up"] for r in rows_m], "ro-", lw=1.4, ms=4, label=r"$D_x$, $\uparrow$")
    ax2.plot(ms, [r["Dx_dn"] for r in rows_m], "bo-", lw=1.4, ms=4, label=r"$D_x$, $\downarrow$")
    ax2.plot(ms, [r["Dx_tot"] for r in rows_m], "k--o", lw=1.2, ms=3, label=r"$D_x$, tot")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$M$ (meV)")
    ax2.set_ylabel(r"$D_x$ (BCD)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Berry curvature map at A0=0.06
    At06, _ = floquet_params(0.06)
    omap = np.zeros_like(kx)
    for i in range(NK):
        for j in range(NK):
            omap[i, j] = (omega_analytic(kx[i, j], ky[i, j], "up", At06, 0.06)
                          - omega_analytic(kx[i, j], ky[i, j], "dn", At06, 0.06))
    vmax = np.percentile(np.abs(omap), 99.0)
    im = ax3.imshow(omap.T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_xlabel(r"$k_x$")
    ax3.set_ylabel(r"$k_y$")
    ax3.set_title(r"$\Omega_\uparrow-\Omega_\downarrow$")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    fig.suptitle(rf"Paper H -- nonlinear Hall (BCD), $M={M0:+.0f}$ meV, $J_0={J0:.0f}$ meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figs/bcd_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/bcd_floquet.json")
    print("Saved: figs/bcd_panel.png")


if __name__ == "__main__":
    main()

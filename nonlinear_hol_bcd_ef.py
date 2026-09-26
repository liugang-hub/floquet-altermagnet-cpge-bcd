# -*- coding: utf-8 -*-
"""
Paper H -- Script 7: BCD vs Fermi level -- nonlinear Hall needs a Fermi
surface.  Topology: for an insulator (f_0 = 0 or 1 everywhere),
D_a = int d^2k f_0 d_a Omega = int d^2k d_a Omega = 0 (total derivative
over the periodic BZ), independent of how anisotropic Omega is.
The BCD is therefore a Fermi-surface property: the Fermi level must
enter the band.  Two ingredients are then needed:
  (i)   a Fermi surface at EF within the band;
  (ii)  mirror breaking (tilt T sin kx, or the real hexagonal lattice of
        MnTe/CrSb) so that f_0(k) is not even in kx.
Floquet drive A0 renormalizes the bands and tunes the dipole.

Outputs:
  - data/bcd_ef.json
  - figs/bcd_ef_panel.png
"""
import os, json, time
import numpy as np

A, B, D, M0 = 364.5, -686.0, -512.0, -10.0
HW = 150.0
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "100"))
DK = 1e-3


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
    d_norm = max(np.linalg.norm(d_vec), 1e-12)
    return np.dot(d_vec, np.cross(dx, dy)) / (2.0 * d_norm**3)


def ev_k(kx, ky, spin, At, A0, T):
    """Valence band energy with tilt."""
    s = S3[spin]
    sinx, cosx = np.sin(kx), np.cos(kx)
    siny, cosy = np.sin(ky), np.cos(ky)
    k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
    h0 = -D * (A0**2 + k2) + T * sinx
    hz = M0 - B * k2 + 2.0 * s * J0 * (cosy - cosx)
    hx = At[spin] * sinx
    hy = At[spin] * siny
    d_norm = np.sqrt(hx**2 + hy**2 + hz**2)
    return h0 - d_norm


def bcd_at_ef(kx, ky, spin, At, A0, T, EF):
    nk = kx.shape[0]
    d2k = (2.0 * np.pi / nk) ** 2
    Dx = Dy = 0.0
    for i in range(nk):
        for j in range(nk):
            kx0, ky0 = kx[i, j], ky[i, j]
            if ev_k(kx0, ky0, spin, At, A0, T) >= EF:
                continue
            Dx += (omega_analytic(kx0 + DK, ky0, spin, At, A0)
                   - omega_analytic(kx0 - DK, ky0, spin, At, A0)) / (2.0 * DK) * d2k
            Dy += (omega_analytic(kx0, ky0 + DK, spin, At, A0)
                   - omega_analytic(kx0, ky0 - DK, spin, At, A0)) / (2.0 * DK) * d2k
    return Dx, Dy


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    T = 30.0
    A0_ef = 0.06
    At_ef, Mt_ef = floquet_params(A0_ef)
    # band edges (valence band top at k=0)
    eup0 = ev_k(0.0, 0.0, "up", At_ef, A0_ef, T)
    edn0 = ev_k(0.0, 0.0, "dn", At_ef, A0_ef, T)
    print("== EF scan, A0=%.2f, T=%.0f meV ==" % (A0_ef, T))
    print("  valence-band top at k=0: up = %+.3f meV, dn = %+.3f meV" % (eup0, edn0))

    ef_list = [-16.0, -12.0, -8.0, -6.0, -4.0, -2.0, 0.0, +2.0, +4.0, +8.0, +12.0, +16.0]
    rows_ef = []
    for EF in ef_list:
        r = {"EF": EF}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_at_ef(kx, ky, spin, At_ef, A0_ef, T, EF)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_ef.append(r)
        print("  EF=%+5.1f: Dx(up,dn,tot)=(%+.4e,%+.4e,%+.4e)  Dy_tot=%+.4e"
              % (EF, r["Dx_up"], r["Dx_dn"], r["Dx_tot"], r["Dy_tot"]))

    # ---- A0 scan at a metallic EF ----
    EF_metal = -4.0   # inside both valence bands (up top -8.9, dn top -2.5)
    print("\n== A0 scan at EF = %+.1f meV, T = %.0f meV ==" % (EF_metal, T))
    a0_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    rows_a0 = []
    for A0 in a0_list:
        At, Mt = floquet_params(A0)
        r = {"A0": A0, "Mt_up": Mt["up"], "Mt_dn": Mt["dn"]}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_at_ef(kx, ky, spin, At, A0, T, EF_metal)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_a0.append(r)
        print("  A0=%.2f: Dx(up,dn,tot)=(%+.4e,%+.4e,%+.4e)  Dy_tot=%+.4e"
              % (A0, r["Dx_up"], r["Dx_dn"], r["Dx_tot"], r["Dy_tot"]))

    out = {
        "params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
                   "NK": NK, "DK": DK, "T": T, "EF_metal_A0scan": EF_metal,
                   "note": "D_a = int d2k f(EF) d_a Omega; insulator limit gives 0 by topology"},
        "ef_scan": rows_ef,
        "a0_scan": rows_a0,
    }
    with open("data/bcd_ef.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 3.6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    efs = [r["EF"] for r in rows_ef]
    ax1.plot(efs, [r["Dx_up"] for r in rows_ef], "ro-", lw=1.4, ms=4, label=r"$D_x$, $\uparrow$")
    ax1.plot(efs, [r["Dx_dn"] for r in rows_ef], "bo-", lw=1.4, ms=4, label=r"$D_x$, $\downarrow$")
    ax1.plot(efs, [r["Dx_tot"] for r in rows_ef], "k--o", lw=1.2, ms=3, label=r"$D_x$, tot")
    ax1.axvline(eup0, color="r", ls=":", lw=0.8)
    ax1.axvline(edn0, color="b", ls=":", lw=0.8)
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel(r"$E_F$ (meV)")
    ax1.set_ylabel(r"$D_x$ (BCD)")
    ax1.set_title(rf"$A_0={A0_ef:.2f}$, $T={T:.0f}$ meV")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    a0s = [r["A0"] for r in rows_a0]
    ax2.plot(a0s, [r["Dx_up"] for r in rows_a0], "ro-", lw=1.4, ms=4, label=r"$D_x$, $\uparrow$")
    ax2.plot(a0s, [r["Dx_dn"] for r in rows_a0], "bo-", lw=1.4, ms=4, label=r"$D_x$, $\downarrow$")
    ax2.plot(a0s, [r["Dx_tot"] for r in rows_a0], "k--o", lw=1.2, ms=3, label=r"$D_x$, tot")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax2.set_ylabel(r"$D_x$ (BCD)")
    ax2.set_title(rf"$E_F={EF_metal:+.0f}$ meV, $T={T:.0f}$ meV")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Fermi-surface occupation map (up) at EF=-4
    At06, _ = floquet_params(A0_ef)
    occ = np.zeros_like(kx)
    for i in range(NK):
        for j in range(NK):
            occ[i, j] = 1.0 if ev_k(kx[i, j], ky[i, j], "up", At06, A0_ef, T) < EF_metal else 0.0
    im = ax3.imshow(occ.T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                    cmap="binary", aspect="auto")
    ax3.set_xlabel(r"$k_x$")
    ax3.set_ylabel(r"$k_y$")
    ax3.set_title(rf"occupied region, $\uparrow$, $E_F={EF_metal:+.0f}$ meV")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    fig.suptitle(rf"Paper H -- BCD vs Fermi level, $M={M0:+.0f}$ meV, $J_0={J0:.0f}$ meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figs/bcd_ef_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/bcd_ef.json")
    print("Saved: figs/bcd_ef_panel.png")


if __name__ == "__main__":
    main()

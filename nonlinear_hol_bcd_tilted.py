# -*- coding: utf-8 -*-
"""
Paper H -- Script 6: nonlinear Hall effect with tilt/strain mirror breaking.

In the square-lattice model without mirror breaking, D_x and D_y vanish
strictly because Omega_s(kx,ky) is even in kx and ky separately, so
d_a Omega_s is odd and integrates to zero over the BZ.  Real MnTe/CrSb
do not have this mirror symmetry; here we model the breaking with a tilt
term in the kinetic energy

      h0 -> h0 + T * sin(kx),

which makes the valence-band occupation f_0(k) asymmetric in kx, turning on
a finite BCD even though Omega_s itself remains mirror-even.  The drive A0
then tunes the BCD through the Floquet-renormalized bands.

Outputs:
  - data/bcd_tilted.json
  - figs/bcd_tilted_panel.png
"""
import os, json, time
import numpy as np

A, B, D, M0 = 364.5, -686.0, -512.0, -10.0
HW = 150.0
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "120"))
EF = 0.0
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


def fermi_factor(kx, ky, spin, At, A0, T):
    s = S3[spin]
    sinx, cosx = np.sin(kx), np.cos(kx)
    siny, cosy = np.sin(ky), np.cos(ky)
    k2 = 2.0 * (1.0 - cosx) + 2.0 * (1.0 - cosy)
    h0 = -D * (A0**2 + k2) + T * sinx
    hz = M0 - B * k2 + 2.0 * s * J0 * (cosy - cosx)
    hx = At[spin] * sinx
    hy = At[spin] * siny
    d_norm = np.sqrt(hx**2 + hy**2 + hz**2)
    return 1.0 if (h0 - d_norm) < EF else 0.0


def bcd_tilted(kx, ky, spin, At, A0, T):
    nk = kx.shape[0]
    d2k = (2.0 * np.pi / nk) ** 2
    Dx = Dy = 0.0
    for i in range(nk):
        for j in range(nk):
            kx0, ky0 = kx[i, j], ky[i, j]
            if fermi_factor(kx0, ky0, spin, At, A0, T) == 0:
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

    # ---- A0 scan at T = 30 meV ----
    T_fixed = 30.0
    print("== A0 scan, M = %+.0f meV, J0 = %.0f meV, T = %.0f meV ==" % (M0, J0, T_fixed))
    a0_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    rows_a0 = []
    for A0 in a0_list:
        At, Mt = floquet_params(A0)
        r = {"A0": A0, "Mt_up": Mt["up"], "Mt_dn": Mt["dn"]}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_tilted(kx, ky, spin, At, A0, T_fixed)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_a0.append(r)
        print("  A0=%.2f: Dx(up,dn,tot)=(%+.3e,%+.3e,%+.3e)  Dy=(%+.3e,%+.3e,%+.3e)"
              % (A0, r["Dx_up"], r["Dx_dn"], r["Dx_tot"],
                 r["Dy_up"], r["Dy_dn"], r["Dy_tot"]))

    # ---- T scan at A0 = 0.06 ----
    A0_fixed = 0.06
    print("\n== T scan, M = %+.0f meV, J0 = %.0f meV, A0 = %.2f ==" % (M0, J0, A0_fixed))
    t_list = [0, 10, 20, 30, 40, 50]
    rows_t = []
    for T in t_list:
        At, Mt = floquet_params(A0_fixed)
        r = {"T": T}
        for spin in ("up", "dn"):
            Dx, Dy = bcd_tilted(kx, ky, spin, At, A0_fixed, T)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        rows_t.append(r)
        print("  T=%.0f: Dx(up,dn,tot)=(%+.3e,%+.3e,%+.3e)  Dy=(%+.3e,%+.3e,%+.3e)"
              % (T, r["Dx_up"], r["Dx_dn"], r["Dx_tot"],
                 r["Dy_up"], r["Dy_dn"], r["Dy_tot"]))

    out = {
        "params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
                   "NK": NK, "EF": EF, "DK": DK, "T_fixed_A0scan": T_fixed,
                   "note": "tilt h0 -> h0 + T sin(kx) breaks kx mirror; D_a = int f d_a Omega"},
        "a0_scan": rows_a0,
        "t_scan": rows_t,
    }
    with open("data/bcd_tilted.json", "w") as f:
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
    ax1.set_title(rf"$T={T_fixed:.0f}$ meV")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ts = [r["T"] for r in rows_t]
    ax2.plot(ts, [r["Dx_up"] for r in rows_t], "ro-", lw=1.4, ms=4, label=r"$D_x$, $\uparrow$")
    ax2.plot(ts, [r["Dx_dn"] for r in rows_t], "bo-", lw=1.4, ms=4, label=r"$D_x$, $\downarrow$")
    ax2.plot(ts, [r["Dx_tot"] for r in rows_t], "k--o", lw=1.2, ms=3, label=r"$D_x$, tot")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$T$ (meV)")
    ax2.set_ylabel(r"$D_x$ (BCD)")
    ax2.set_title(rf"$A_0={A0_fixed:.3f}$ nm$^{{-1}}$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # occupation asymmetry at A0=0.06, T=30
    At06, _ = floquet_params(A0_fixed)
    nup = np.zeros_like(kx)
    ndn = np.zeros_like(kx)
    for i in range(NK):
        for j in range(NK):
            nup[i, j] = fermi_factor(kx[i, j], ky[i, j], "up", At06, A0_fixed, T_fixed)
            ndn[i, j] = fermi_factor(kx[i, j], ky[i, j], "dn", At06, A0_fixed, T_fixed)
    im = ax3.imshow((nup - ndn).T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi],
                    cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xlabel(r"$k_x$")
    ax3.set_ylabel(r"$k_y$")
    ax3.set_title(r"$f_\uparrow - f_\downarrow$")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    fig.suptitle(rf"Paper H -- BCD with mirror-breaking tilt, $M={M0:+.0f}$ meV, $J_0={J0:.0f}$ meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figs/bcd_tilted_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/bcd_tilted.json")
    print("Saved: figs/bcd_tilted_panel.png")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper H -- Script 3: A0 fine scan of the spin-resolved CPGE at the
working point M=-10 meV, altermagnet J0=160 meV.

Physics question:
  How does the drive amplitude A0 turn on the spin polarization of the
  circular photocurrent?  At A0=0 the C4 x spin-rotation symmetry forces
  beta_up = beta_dn (P_cpge = 0).  Floquet renormalization breaks this
  symmetry through A_t^up != A_t^dn, and we quantify:
    - the split beta_up(A0) / beta_dn(A0) at the total-CPGE peak
    - P_cpge at the peak, P_cpge max over omega, and the integrated
      low-frequency polarization  P_int = (I_up - I_dn)/(|I_up|+|I_dn|)
      with I_s = int_{1..40 meV} beta_s(omega) domega

Outputs:
  - data/cpge_a0scan.json
  - figs/cpge_a0scan_panel.png
"""
import os, json, time
import numpy as np

# ---------------------------- base parameters -------------------------------
A, B, D, M0 = 364.5, -686.0, -512.0, -10.0    # meV, nm
HW = 150.0
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))

NK   = int(os.environ.get("PH_NK", "160"))
ETA  = float(os.environ.get("PH_ETA", "4.0"))
NW   = int(os.environ.get("PH_NW", "200"))
EF   = 0.0
A0_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]

M_CURRENT = M0
OM_MIN, OM_MAX = 1.0, 60.0


def floquet_params(A0, sp=SP):
    At = {
        "up": (1.0 - 2.0 * A0**2 * B * sp / HW) * A,
        "dn": (1.0 + 2.0 * A0**2 * B * sp / HW) * A,
    }
    Mt = {
        "up": M0 - B * A0**2 - A0**2 * A**2 * sp / HW,
        "dn": M0 - B * A0**2 + A0**2 * A**2 * sp / HW,
    }
    Eref = -D * A0**2
    return At, Mt, Eref


def bulk_hamiltonian(kx, ky, spin, At):
    k2 = 2.0 * (1.0 - np.cos(kx)) + 2.0 * (1.0 - np.cos(ky))
    m = M_CURRENT - B * k2 + 2.0 * S3[spin] * J0 * (np.cos(ky) - np.cos(kx))
    h0 = -D * (At["A0"]**2 + k2)
    hx = At[spin] * np.sin(kx)
    hy = At[spin] * np.sin(ky)
    hz = m
    return np.array([[h0 + hz, hx - 1j * hy],
                     [hx + 1j * hy, h0 - hz]], dtype=complex)


def deriv_hamiltonian(kx, ky, spin, At, axis):
    if axis == 0:
        sx = 2.0 * np.sin(kx) * (B + S3[spin] * J0)
        s0 = 2.0 * D * np.sin(kx)
        off_re = At[spin] * np.cos(kx)
        off_im = 0.0
    else:
        sx = 2.0 * np.sin(ky) * (B - S3[spin] * J0)
        s0 = 2.0 * D * np.sin(ky)
        off_re = 0.0
        off_im = At[spin] * np.cos(ky)
    return np.array([[s0 + sx, off_re - 1j * off_im],
                     [off_re + 1j * off_im, s0 - sx]], dtype=complex)


def compute_spin_cpge(kx, ky, spin, At, omega, eta=ETA):
    nk = kx.shape[0]
    d2k = (2.0 * np.pi / nk) ** 2
    E = np.zeros((nk, nk, 2), dtype=float)
    U = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i in range(nk):
        for j in range(nk):
            H = bulk_hamiltonian(kx[i, j], ky[i, j], spin, At)
            e, u = np.linalg.eigh(H)
            E[i, j] = e
            U[i, j] = u
    Vx = np.zeros((nk, nk, 2, 2), dtype=complex)
    Vy = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i in range(nk):
        for j in range(nk):
            dHx = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, axis=0)
            dHy = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, axis=1)
            ud = U[i, j].conj().T
            Vx[i, j] = ud @ dHx @ U[i, j]
            Vy[i, j] = ud @ dHy @ U[i, j]
    dE = E[:, :, 1] - E[:, :, 0]
    gap_ok = dE > 1e-6
    dE_safe = np.where(gap_ok, dE, 1.0)
    rx_cv = 1j * Vx[:, :, 0, 1] / dE_safe
    ry_vc = 1j * Vy[:, :, 1, 0] / (-dE_safe)
    weight = np.where(gap_ok, np.imag(rx_cv * ry_vc), 0.0)
    f_diff = np.where((E[:, :, 0] < EF) & (E[:, :, 1] > EF), 1.0, 0.0)
    beta = np.zeros_like(omega, dtype=float)
    for iom, om in enumerate(omega):
        lor = eta / (np.pi * ((om - dE)**2 + eta**2))
        beta[iom] = np.sum(f_diff * weight * lor) * d2k
    return beta


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    omega = np.linspace(OM_MIN, OM_MAX, NW)
    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    # integrate window for I_s
    w_sel = (omega >= 1.0) & (omega <= 40.0)
    d_om = omega[1] - omega[0]

    rows = []
    spectra = {}
    print(f"[M = {M0:+.1f} meV, J0 = {J0:.0f} meV, A0 fine scan]")
    for A0 in A0_LIST:
        At, Mt, Eref = floquet_params(A0)
        At["A0"] = A0
        b_up = compute_spin_cpge(kx, ky, "up", At, omega)
        b_dn = compute_spin_cpge(kx, ky, "dn", At, omega)
        b_tot = b_up + b_dn
        denom = np.abs(b_up) + np.abs(b_dn)
        P = np.where(denom > 1e-12, (b_up - b_dn) / denom, 0.0)

        i_peak = np.argmax(np.abs(b_tot))
        om_peak = omega[i_peak]
        I_up = np.sum(b_up[w_sel]) * d_om
        I_dn = np.sum(b_dn[w_sel]) * d_om
        P_int = (I_up - I_dn) / (np.abs(I_up) + np.abs(I_dn) + 1e-30)
        i_pmax = np.argmax(np.abs(P))
        P_at_peak = P[i_peak]

        rows.append({
            "A0": A0,
            "Mt_up": float(Mt["up"]), "Mt_dn": float(Mt["dn"]),
            "At_up": float(At["up"]), "At_dn": float(At["dn"]),
            "om_peak": float(om_peak),
            "beta_up_peak": float(b_up[i_peak]),
            "beta_dn_peak": float(b_dn[i_peak]),
            "beta_tot_peak": float(b_tot[i_peak]),
            "P_at_peak": float(P_at_peak),
            "P_max": float(P[i_pmax]), "om_Pmax": float(omega[i_pmax]),
            "I_up": float(I_up), "I_dn": float(I_dn),
            "P_int": float(P_int),
        })
        spectra[f"A0_{A0:.2f}"] = {
            "omega": omega.tolist(),
            "beta_up": b_up.tolist(), "beta_dn": b_dn.tolist(),
            "beta_tot": b_tot.tolist(), "P": P.tolist(),
        }
        print(f"  A0={A0:5.2f}: Mt({Mt['up']:+6.2f},{Mt['dn']:+6.2f}) "
              f"peak@={om_peak:5.2f}meV up={b_up[i_peak]:+.4e} dn={b_dn[i_peak]:+.4e} "
              f"P_peak={P_at_peak:+.3f} P_max={P[i_pmax]:+.3f} P_int={P_int:+.3f}")

    out = {
        "params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
                   "A0_list": A0_LIST, "NK": NK, "ETA": ETA, "NW": NW, "EF": EF,
                   "note": "P=(b_up-b_dn)/(|b_up|+|b_dn|); I_s=int_{1..40meV} beta_s domega"},
        "rows": rows,
        "spectra": spectra,
    }
    with open("data/cpge_a0scan.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A0s = np.array([r["A0"] for r in rows])
    fig = plt.figure(figsize=(11, 3.4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # (a) beta_up / beta_dn / beta_tot at peak vs A0
    ax1.plot(A0s, [r["beta_up_peak"] for r in rows], "ro-", lw=1.4, ms=4, label=r"$\uparrow$")
    ax1.plot(A0s, [r["beta_dn_peak"] for r in rows], "bo-", lw=1.4, ms=4, label=r"$\downarrow$")
    ax1.plot(A0s, [r["beta_tot_peak"] for r in rows], "k--o", lw=1.2, ms=3, label="total")
    ax1.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax1.set_ylabel(r"$\beta^{xy}$ at peak (a.u.)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # (b) P_cpge at peak / P_max / P_int vs A0
    ax2.plot(A0s, [r["P_at_peak"] for r in rows], "m^-", lw=1.4, ms=5, label=r"$P$ at peak")
    ax2.plot(A0s, [r["P_max"] for r in rows], "cs-", lw=1.2, ms=4, label=r"$P_{\max}$")
    ax2.plot(A0s, [r["P_int"] for r in rows], "g*-", lw=1.4, ms=5, label=r"$P_{\rm int}$")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax2.set_ylabel("photocurrent spin polarization")
    ax2.set_ylim(-1.1, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (c) representative P(omega) spectra
    cmap = plt.get_cmap("viridis")
    for iA, A0 in enumerate(A0_LIST):
        if A0 in (0.0, 0.06, 0.12):
            dat = spectra[f"A0_{A0:.2f}"]
            ax3.plot(dat["omega"], dat["P"], color=cmap(A0 / 0.16), lw=1.4,
                     label=rf"$A_0={A0:.2f}$")
    ax3.axhline(0, color="gray", ls=":", lw=0.8)
    ax3.set_xlabel(r"$\hbar\omega$ (meV)")
    ax3.set_ylabel(r"$P_{\rm cpge}(\omega)$")
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(rf"Paper H -- A0 scan, $M={M0:+.0f}$ meV, $J_0={J0:.0f}$ meV: optical control of photocurrent spin polarization",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figs/cpge_a0scan_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/cpge_a0scan.json")
    print("Saved: figs/cpge_a0scan_panel.png")


if __name__ == "__main__":
    main()

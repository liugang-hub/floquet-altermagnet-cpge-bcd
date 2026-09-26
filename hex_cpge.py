# -*- coding: utf-8 -*-
"""
Paper H -- Script 9 (hexagonal): spin-resolved CPGE on the hexagonal lattice.

Continuous k.p BHZ model on the TRUE hexagonal 1st BZ of the triangular
lattice (see hex_model.py, v2).  Compared to the square-lattice Script 1/2:

  - integration domain: hexagonal BZ (R = 4pi/3), 6-edge mask sampling;
  - d-wave altermagnet with MnTe (Cm'c'm) orientation:
        fd = (3/8)(ky^2 - kx^2),   m_s = Mt_s - B k^2 + 2 s J0 fd;
  - tilt T kx in h0 (mirror breaking, keeps the CPGE/BCD setups consistent);
  - Mt_s is really used in the mass term (Script 2 computed but ignored it).

Symmetry note: on the hexagonal lattice NO C6v operation maps fd -> -fd
(the square lattice had the 90-degree rotation; C6v does not), so beta_up =
beta_dn is NOT exactly protected: at A0=0 the main peak (Gamma transitions)
still shows P ~ 0 to high accuracy, while the high-frequency tail carries a
small lattice-scale background.  The Floquet drive splits At_s / Mt_s and
turns P_peak into a large, continuously tunable signal.

Outputs:
  - data/hex_cpge.json
  - figs/hex_cpge_panel.png
  - terminal summary
"""
import os, json, time
import numpy as np
import hex_model as hm

# ------------------------------ parameters ---------------------------------
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
HW = hm.HW_PARAM
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
TILT = float(os.environ.get("PH_TILT", "30.0"))

M_LIST = np.array([float(x) for x in os.environ.get("PH_M", "-10.0").split(",")])
A0_LIST = [0.0, 0.06, 0.12]
NK = int(os.environ.get("PH_NK", "160"))
ETA = float(os.environ.get("PH_ETA", "4.0"))
NW = int(os.environ.get("PH_NW", "120"))
EF = 0.0
M_CURRENT = M0


def floquet_params(A0, sp=SP):
    At = {
        "up": (1.0 - 2.0 * A0**2 * B * sp / HW) * A,
        "dn": (1.0 + 2.0 * A0**2 * B * sp / HW) * A,
    }
    Mt = {
        "up": M_CURRENT - B * A0**2 - A0**2 * A**2 * sp / HW,
        "dn": M_CURRENT - B * A0**2 + A0**2 * A**2 * sp / HW,
    }
    return At, Mt


def bulk_hamiltonian(kx, ky, spin, At, Mt):
    k2 = kx * kx + ky * ky
    fd = 0.375 * (ky * ky - kx * kx)
    s = S3[spin]
    m = Mt[spin] - B * k2 + 2.0 * s * J0 * fd
    h0 = -D * At["A0"]**2 - D * k2 + TILT * kx
    hx = At[spin] * kx
    hy = At[spin] * ky
    return np.array([[h0 + m, hx - 1j * hy],
                     [hx + 1j * hy, h0 - m]], dtype=complex)


def deriv_hamiltonian(kx, ky, spin, At, Mt, axis):
    s = S3[spin]
    if axis == 0:
        dm = -2.0 * B * kx - 1.5 * s * J0 * kx   # d/dkx of 2sJ0*fd (FIX v2.1: was 0.75)
        dh0 = -2.0 * D * kx + TILT
        dhx = At[spin]
        dhy = 0.0
    else:
        dm = -2.0 * B * ky + 1.5 * s * J0 * ky   # d/dky of 2sJ0*fd (FIX v2.1: was 0.75)
        dh0 = -2.0 * D * ky
        dhx = 0.0
        dhy = At[spin]
    return np.array([[dh0 + dm, dhx - 1j * dhy],
                     [dhx + 1j * dhy, dh0 - dm]], dtype=complex)


def compute_spin_cpge(kx, ky, mask, d2k, spin, At, Mt, omega, eta=ETA):
    nk = kx.shape[0]
    idx = np.argwhere(mask)
    E = np.zeros((nk, nk, 2), dtype=float)
    U = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i, j in idx:
        H = bulk_hamiltonian(kx[i, j], ky[i, j], spin, At, Mt)
        e, u = np.linalg.eigh(H)
        E[i, j] = e
        U[i, j] = u

    Vx = np.zeros((nk, nk, 2, 2), dtype=complex)
    Vy = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i, j in idx:
        dHx = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, Mt, axis=0)
        dHy = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, Mt, axis=1)
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
        beta[iom] = np.sum(f_diff * weight * lor * mask) * d2k
    return beta


def main():
    global M_CURRENT
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    omega = np.linspace(1.0, 60.0, NW)
    kx, ky, mask, d2k = hm.hex_mesh(NK)
    results = {}
    summary = []

    for M_scan in M_LIST:
        M_CURRENT = M_scan
        key_M = f"M{M_scan:+.1f}"
        results[key_M] = {}
        print(f"\n[M = {M_scan:+.1f} meV, J0 = {J0:.0f}, TILT = {TILT:.0f}, A = {A:.0f}]")
        for A0 in A0_LIST:
            At, Mt = floquet_params(A0)
            At["A0"] = A0
            b_up = compute_spin_cpge(kx, ky, mask, d2k, "up", At, Mt, omega)
            b_dn = compute_spin_cpge(kx, ky, mask, d2k, "dn", At, Mt, omega)
            b_tot = b_up + b_dn
            denom = np.abs(b_up) + np.abs(b_dn)
            P_cpge = np.where(denom > 1e-12, (b_up - b_dn) / denom, 0.0)

            key_A0 = f"A0_{A0:.3f}"
            results[key_M][key_A0] = {
                "omega": omega.tolist(),
                "beta_up": b_up.tolist(),
                "beta_dn": b_dn.tolist(),
                "beta_tot": b_tot.tolist(),
                "P_cpge": P_cpge.tolist(),
            }

            i_up = np.argmax(np.abs(b_up)); i_dn = np.argmax(np.abs(b_dn))
            i_tot = np.argmax(np.abs(b_tot)); i_p = np.argmax(np.abs(P_cpge))
            summary.append({
                "M": float(M_scan), "A0": A0, "J0": J0, "TILT": TILT, "A": A,
                "peak_up": [float(omega[i_up]), float(b_up[i_up])],
                "peak_dn": [float(omega[i_dn]), float(b_dn[i_dn])],
                "peak_tot": [float(omega[i_tot]), float(b_tot[i_tot])],
                "sign_tot": int(np.sign(b_tot[i_tot])),
                "P_cpge_max": [float(omega[i_p]), float(P_cpge[i_p])],
            })
            print(f"  A0={A0:5.3f}: up@={omega[i_up]:5.2f}meV({b_up[i_up]:+.3e}) "
                  f"dn@={omega[i_dn]:5.2f}meV({b_dn[i_dn]:+.3e}) "
                  f"tot={int(np.sign(b_tot[i_tot])):+d}@={omega[i_tot]:5.2f}meV "
                  f"P_cpge_max={P_cpge[i_p]:+.3f}")

    # Chern numbers at the working point
    print("\n[Chern numbers on the hexagonal BZ, A0 = 0]")
    At0, Mt0 = floquet_params(0.0)
    chern = {}
    for spin in ("up", "dn"):
        C = hm.chern_number(kx, ky, mask, d2k, spin, At0, Mt0, 0.0, 0.0, J0)
        chern[spin] = C
        print(f"  C_{spin} = {C:+.3f}")

    out = {
        "params": {
            "A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
            "TILT": TILT, "M_list": M_LIST.tolist(), "A0_list": A0_LIST,
            "NK": NK, "ETA": ETA, "NW": NW, "EF": EF,
            "lattice": "triangular, hexagonal 1st BZ, R = 4pi/3",
            "model": "continuous k.p + hex BZ (v2); fd=(3/8)(ky^2-kx^2) MnTe Cm'c'm",
            "note": "energy meV; beta normalized ~ e^3/hbar^2; P=(b_up-b_dn)/(|b_up|+|b_dn|)"
        },
        "summary": summary,
        "chern": chern,
        "results": results,
    }
    with open("data/hex_cpge.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nM = len(M_LIST)
    nA = len(A0_LIST)
    fig, axes = plt.subplots(nM, nA, figsize=(3.2 * nA + 0.6, 2.6 * nM + 0.6), sharex=True, sharey=False)
    if nM == 1:
        axes = axes.reshape(1, -1)
    for iM, M_scan in enumerate(M_LIST):
        for iA, A0 in enumerate(A0_LIST):
            ax = axes[iM, iA]
            dat = results[f"M{M_scan:+.1f}"][f"A0_{A0:.3f}"]
            om = np.array(dat["omega"])
            ax.plot(om, np.array(dat["beta_up"]) * 1e3, "r-", lw=1.2, label=r"$\uparrow$")
            ax.plot(om, np.array(dat["beta_dn"]) * 1e3, "b-", lw=1.2, label=r"$\downarrow$")
            ax.plot(om, np.array(dat["beta_tot"]) * 1e3, "k--", lw=1.5, label="total")
            ax2 = ax.twinx()
            ax2.plot(om, np.array(dat["P_cpge"]), "g:", lw=1.0, alpha=0.8)
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("$P_{\\rm cpge}$", fontsize=8, color="green")
            ax.axhline(0, color="gray", ls=":", lw=0.6)
            ax.set_xlim(0, 60)
            if iM == nM - 1:
                ax.set_xlabel(r"$\hbar\omega$ (meV)")
            if iA == 0:
                ax.set_ylabel(r"$\beta^{xy}\,(\times10^{-3})$")
            if iM == 0:
                ax.set_title(f"$A_0={A0:.3f}$ nm$^{{-1}}$")
            if iA == nA - 1:
                ax.text(1.03, 0.5, f"$M={M_scan:+.0f}$ meV", transform=ax.transAxes,
                        va="center", ha="left", fontsize=10, rotation=90)
            if iM == 0 and iA == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(rf"Paper H -- hexagonal-lattice CPGE, $J_0={J0:.0f}$ meV, $T={TILT:.0f}$ meV",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figs/hex_cpge_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/hex_cpge.json")
    print("Saved: figs/hex_cpge_panel.png")


if __name__ == "__main__":
    main()

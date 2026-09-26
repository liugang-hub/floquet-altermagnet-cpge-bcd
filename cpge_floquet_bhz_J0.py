# -*- coding: utf-8 -*-
"""
Paper H -- Script 2: spin-resolved CPGE with the d-wave altermagnet term.
Script 1 established sign(beta_tot) = sign(Chern number) for the bare BHZ.
Here the altermagnet exchange J0 (d-wave, s_z-block diagonal) is switched on:

    m_s(k) = M - B k^2 + 2 s J0 [cos(ky) - cos(kx)],   s = up/down

This splits the spin blocks in momentum space (spin-split nodal structure),
so the spin-resolved photocurrents differ: beta_up != beta_dn.  We output

    P_cpge(omega) = (beta_up - beta_dn) / (beta_up + beta_dn)

as the "photocurrent spin polarization" fingerprint of the altermagnetic
order, and track how the drive A0 modulates it.

Outputs:
  - data/cpge_floquet_bhz_J0.json
  - figs/cpge_J0_panel.png
  - terminal summary
"""
import os, json, time
import numpy as np

# ---------------------------- base parameters -------------------------------
A, B, D, M0 = 364.5, -686.0, -512.0, -10.0    # meV, nm
HW = 150.0                                     # meV photon energy
SP = 1.0                                       # helicity: +1 = sigma+
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))   # meV, d-wave altermagnet term

# ------------------------------- scan knobs ---------------------------------
M_LIST   = np.array([float(x) for x in os.environ.get("PH_M", "-10.0").split(",")])
A0_LIST  = [0.0, 0.06, 0.12]                   # nm^-1
NK       = int(os.environ.get("PH_NK", "160"))
ETA      = float(os.environ.get("PH_ETA", "4.0"))
NW       = int(os.environ.get("PH_NW", "120"))
EF       = 0.0

M_CURRENT = M0

# --------------------------- Floquet renormalization ------------------------
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


# ------------------------- 2x2 bulk Hamiltonian -----------------------------
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
    if axis == 0:  # x
        sx = 2.0 * np.sin(kx) * (B + S3[spin] * J0)
        s0 = 2.0 * D * np.sin(kx)
        off_re = At[spin] * np.cos(kx)
        off_im = 0.0
    else:          # y
        sx = 2.0 * np.sin(ky) * (B - S3[spin] * J0)
        s0 = 2.0 * D * np.sin(ky)
        off_re = 0.0
        off_im = At[spin] * np.cos(ky)
    return np.array([[s0 + sx, off_re - 1j * off_im],
                     [off_re + 1j * off_im, s0 - sx]], dtype=complex)


# ----------------------------- CPGE kernel ----------------------------------
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


# --------------------------------- main -------------------------------------
def main():
    global M_CURRENT
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    omega = np.linspace(1.0, 60.0, NW)
    results = {}
    summary = []

    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    for M_scan in M_LIST:
        M_CURRENT = M_scan
        key_M = f"M{M_scan:+.1f}"
        results[key_M] = {}
        print(f"\n[M = {M_scan:+.1f} meV, J0 = {J0:.0f} meV]")
        for A0 in A0_LIST:
            At, Mt, Eref = floquet_params(A0)
            At["A0"] = A0
            b_up = compute_spin_cpge(kx, ky, "up", At, omega)
            b_dn = compute_spin_cpge(kx, ky, "dn", At, omega)
            b_tot = b_up + b_dn
            # photocurrent spin polarization (guard against near-zero denom)
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
                "M": float(M_scan), "A0": A0, "J0": J0,
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

    out = {
        "params": {
            "A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
            "M_list": M_LIST.tolist(), "A0_list": A0_LIST,
            "NK": NK, "ETA": ETA, "NW": NW, "EF": EF,
            "note": "energy meV; beta in normalized units ~ e^3/hbar^2; P_cpge=(b_up-b_dn)/(|b_up|+|b_dn|)"
        },
        "summary": summary,
        "results": results,
    }
    with open("data/cpge_floquet_bhz_J0.json", "w") as f:
        json.dump(out, f, indent=2)

    # figure: A0 scan for the working point M=-10
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
            key_M = f"M{M_scan:+.1f}"; key_A0 = f"A0_{A0:.3f}"
            dat = results[key_M][key_A0]
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
    fig.suptitle(rf"Paper H -- spin-resolved CPGE, altermagnet $J_0={J0:.0f}$ meV", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figs/cpge_J0_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/cpge_floquet_bhz_J0.json")
    print("Saved: figs/cpge_J0_panel.png")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper H -- Script 1: spin-resolved circular photogalvanic effect (CPGE)
in the Floquet-renormalized bulk BHZ model.

Physics:
  CPGE measures Berry-curvature-weighted interband transitions under
  circularly polarized light.  The photocurrent tensor is

      beta^{xy}(omega) ~ sum_{v,c} int d^2k [f_v - f_c]
                         Im[ r^x_cv(k) r^y_vc(k) ] delta(omega - omega_cv(k))

  with r^a_nm(k) = i <n|partial_a H|m> / (E_m - E_n).

  The Floquet renormalization At^{spin}, Mt^{spin}, Eref is taken from
  Paper A/F/G (alm_gap_diag.py, verified element-wise).

Outputs:
  - data/cpge_floquet_bhz.json
  - figs/cpge_panel.png (M x A0 panel)
  - terminal summary

No kwant dependency; pure numpy + matplotlib.
"""
import os, json, time
import numpy as np

# ---------------------------- base parameters -------------------------------
A, B, D, M0 = 364.5, -686.0, -512.0, -10.0    # meV, nm
HW = 150.0                                     # meV photon energy
SP = 1.0                                       # helicity: +1 = sigma+
S3 = {"up": +1.0, "dn": -1.0}
J0 = 0.0                                       # altermagnet term off in Script 1

# ------------------------------- scan knobs ---------------------------------
M_LIST   = np.array([-30.0, -15.0, -10.0, -5.0, 0.0, +10.0])  # meV
A0_LIST  = [0.0, 0.06, 0.12]                                   # nm^-1
NK       = int(os.environ.get("PE_CPGE_NK", "160"))            # k-grid per direction
ETA      = float(os.environ.get("PE_CPGE_ETA", "4.0"))         # Lorentzian broadening (meV)
NW       = int(os.environ.get("PE_CPGE_NW", "120"))            # omega points
EF       = 0.0                                                 # Fermi level (half filling)

M_CURRENT = M0                                                 # mutated by main loop

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
    """2x2 bulk BHZ Hamiltonian for one spin block, a=1 nm lattice."""
    k2 = 2.0 * (1.0 - np.cos(kx)) + 2.0 * (1.0 - np.cos(ky))
    m = M_CURRENT - B * k2 + 2.0 * S3[spin] * J0 * (np.cos(ky) - np.cos(kx))
    h0 = -D * (At["A0"]**2 + k2)
    hx = At[spin] * np.sin(kx)
    hy = At[spin] * np.sin(ky)
    hz = m
    return np.array([[h0 + hz, hx - 1j * hy],
                     [hx + 1j * hy, h0 - hz]], dtype=complex)


def deriv_hamiltonian(kx, ky, spin, At, axis):
    """Analytic derivative dH/dk_a for the 2x2 bulk Hamiltonian."""
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
    """Return beta^{xy}(omega) for one spin block."""
    nk = kx.shape[0]
    d2k = (2.0 * np.pi / nk) ** 2

    # Precompute eigenvalues/eigenvectors
    E = np.zeros((nk, nk, 2), dtype=float)
    U = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i in range(nk):
        for j in range(nk):
            H = bulk_hamiltonian(kx[i, j], ky[i, j], spin, At)
            e, u = np.linalg.eigh(H)
            E[i, j] = e
            U[i, j] = u

    # Velocity matrix elements
    Vx = np.zeros((nk, nk, 2, 2), dtype=complex)
    Vy = np.zeros((nk, nk, 2, 2), dtype=complex)
    for i in range(nk):
        for j in range(nk):
            dHx = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, axis=0)
            dHy = deriv_hamiltonian(kx[i, j], ky[i, j], spin, At, axis=1)
            ud = U[i, j].conj().T
            Vx[i, j] = ud @ dHx @ U[i, j]
            Vy[i, j] = ud @ dHy @ U[i, j]

    # valence->conduction transition energy and position matrix elements
    dE = E[:, :, 1] - E[:, :, 0]
    # gap-closure guard: transitions with vanishing dE are excluded
    gap_ok = dE > 1e-6
    dE_safe = np.where(gap_ok, dE, 1.0)
    # r^x_{cv} = i v^x_cv / (E_c - E_v)
    rx_cv = 1j * Vx[:, :, 0, 1] / dE_safe
    # r^y_{vc} = i v^y_vc / (E_v - E_c)  [note: vc means v->c matrix element]
    ry_vc = 1j * Vy[:, :, 1, 0] / (-dE_safe)

    weight = np.where(gap_ok, np.imag(rx_cv * ry_vc), 0.0)
    # T=0 occupation: valence below EF and conduction above EF -> transition active
    f_diff = np.where((E[:, :, 0] < EF) & (E[:, :, 1] > EF), 1.0, 0.0)

    # Lorentzian broadening over omega grid (vectorized over k for each omega)
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
        print(f"\n[M = {M_scan:+.1f} meV]")
        for A0 in A0_LIST:
            At, Mt, Eref = floquet_params(A0)
            At["A0"] = A0          # pass A0 into Hamiltonian for Eref shift
            b_up = compute_spin_cpge(kx, ky, "up", At, omega, eta=ETA)
            b_dn = compute_spin_cpge(kx, ky, "dn", At, omega, eta=ETA)
            b_tot = b_up + b_dn
            key_A0 = f"A0_{A0:.3f}"
            results[key_M][key_A0] = {
                "omega": omega.tolist(),
                "beta_up": b_up.tolist(),
                "beta_dn": b_dn.tolist(),
                "beta_tot": b_tot.tolist(),
            }
            # simple summary: peak magnitude and sign
            idx_peak = np.argmax(np.abs(b_tot))
            om_peak = omega[idx_peak]
            sign_peak = np.sign(b_tot[idx_peak])
            summary.append({
                "M": float(M_scan), "A0": A0,
                "peak_omega": float(om_peak),
                "peak_beta_tot": float(b_tot[idx_peak]),
                "sign": int(sign_peak),
            })
            print(f"  A0={A0:5.3f}: |beta_tot|_max={np.max(np.abs(b_tot)):12.5e} "
                  f"@ omega={om_peak:5.2f} meV  sign={int(sign_peak):+d}")

    # save json
    out = {
        "params": {
            "A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP,
            "M_list": M_LIST.tolist(),
            "A0_list": A0_LIST,
            "J0": J0, "NK": NK, "ETA": ETA, "NW": NW, "EF": EF,
            "note": "units: energy in meV; beta in arbitrary normalized units proportional to e^3/hbar^2"
        },
        "summary": summary,
        "results": results,
    }
    with open("data/cpge_floquet_bhz.json", "w") as f:
        json.dump(out, f, indent=2)

    # figure: M x A0 panel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nM = len(M_LIST)
    nA = len(A0_LIST)
    fig, axes = plt.subplots(nM, nA, figsize=(3.0 * nA + 0.6, 2.0 * nM + 0.6), sharex=True, sharey=True)
    if nM == 1:
        axes = axes.reshape(1, -1)
    for iM, M_scan in enumerate(M_LIST):
        for iA, A0 in enumerate(A0_LIST):
            ax = axes[iM, iA]
            key_M = f"M{M_scan:+.1f}"
            key_A0 = f"A0_{A0:.3f}"
            dat = results[key_M][key_A0]
            om = np.array(dat["omega"])
            ax.plot(om, np.array(dat["beta_up"]) * 1e3, "r-", lw=1.2, label=r"$\uparrow$")
            ax.plot(om, np.array(dat["beta_dn"]) * 1e3, "b-", lw=1.2, label=r"$\downarrow$")
            ax.plot(om, np.array(dat["beta_tot"]) * 1e3, "k--", lw=1.5, label="total")
            ax.axvline(x=2.0 * abs(M_scan), color="gray", ls=":", lw=0.8, alpha=0.6)
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
    fig.suptitle("Paper H -- spin-resolved CPGE of Floquet-renormalized bulk BHZ", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("figs/cpge_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/cpge_floquet_bhz.json")
    print("Saved: figs/cpge_panel.png")


if __name__ == "__main__":
    main()

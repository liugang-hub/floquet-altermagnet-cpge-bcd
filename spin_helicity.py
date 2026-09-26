# -*- coding: utf-8 -*-
"""
Paper H -- Script 8: helicity dependence of the spin-resolved CPGE.

The Floquet renormalization depends on the pump helicity sp:
    At^{up}(sp) = At^{dn}(-sp),  Mt^{up}(sp) = Mt^{dn}(-sp).
Consequently the spin-resolved CPGE obeys the helicity-spin locking
    beta^{up}(sp=+1, omega) = beta^{dn}(sp=-1, omega),
and the total photocurrent spin polarization flips with helicity:
    P_cpge(sp=-1) = -P_cpge(sp=+1).
This is the direct optical counterpart of the sp -> -sp switching in
Papers F/G and is the key experimental fingerprint (measurable by
flipping the pump handedness).

Outputs:
  - data/cpge_helicity.json
  - figs/cpge_helicity_panel.png
"""
import os, json, time
import numpy as np

import cpge_floquet_bhz_J0 as core   # reuse Script 2 kernels

A, B, D, M0 = 364.5, -686.0, -512.0, -10.0
HW = 150.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "160"))
ETA = 4.0
NW = int(os.environ.get("PH_NW", "200"))
EF = 0.0
M_CURRENT = M0


def compute_all(A0, sp, omega, kx, ky):
    """beta_up/dn/tot + P for one (A0, sp)."""
    At, Mt, Eref = core.floquet_params(A0, sp=sp)
    At["A0"] = A0
    core.M_CURRENT = M_CURRENT
    b_up = core.compute_spin_cpge(kx, ky, "up", At, omega, eta=ETA)
    b_dn = core.compute_spin_cpge(kx, ky, "dn", At, omega, eta=ETA)
    b_tot = b_up + b_dn
    denom = np.abs(b_up) + np.abs(b_dn)
    P = np.where(denom > 1e-12, (b_up - b_dn) / denom, 0.0)
    return b_up, b_dn, b_tot, P


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    omega = np.linspace(1.0, 60.0, NW)
    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    print(f"[M={M0:+.0f} meV, J0={J0:.0f} meV, helicity scan]")
    store = {}
    rows = []
    for A0 in (0.06, 0.12):
        for sp in (+1.0, -1.0):
            b_up, b_dn, b_tot, P = compute_all(A0, sp, omega, kx, ky)
            key = f"A0_{A0:.2f}_sp{sp:+.0f}"
            store[key] = {
                "omega": omega.tolist(),
                "beta_up": b_up.tolist(),
                "beta_dn": b_dn.tolist(),
                "beta_tot": b_tot.tolist(),
                "P": P.tolist(),
            }
            i_peak = np.argmax(np.abs(b_tot))
            denom = np.abs(b_up) + np.abs(b_dn)
            P_peak = (b_up[i_peak] - b_dn[i_peak]) / (denom[i_peak] + 1e-30)
            i_pmax = np.argmax(np.abs(P))
            rows.append({
                "A0": A0, "sp": sp,
                "om_peak": float(omega[i_peak]),
                "beta_up_peak": float(b_up[i_peak]),
                "beta_dn_peak": float(b_dn[i_peak]),
                "beta_tot_peak": float(b_tot[i_peak]),
                "P_peak": float(P_peak),
                "P_max": float(P[i_pmax]),
            })
            print("  A0=%.2f sp=%+.0f: tot@=%5.2fmeV(%+.4e)  up=%+.4e dn=%+.4e  P_peak=%+.4f  P_max=%+.4f"
                  % (A0, sp, omega[i_peak], b_tot[i_peak], b_up[i_peak], b_dn[i_peak],
                     P_peak, P[i_pmax]))

    # helicity-spin locking check
    print("\n  Locking check  beta_up(sp=+1) == beta_dn(sp=-1):")
    for A0 in (0.06, 0.12):
        b_up_p = np.array(store[f"A0_{A0:.2f}_sp+1"]["beta_up"])
        b_dn_m = np.array(store[f"A0_{A0:.2f}_sp-1"]["beta_dn"])
        err = np.max(np.abs(b_up_p - b_dn_m))
        print(f"    A0={A0:.2f}: max|beta_up(sp+1) - beta_dn(sp-1)| = {err:.3e}")

    out = {"params": {"M": M0, "J0": J0, "NK": NK, "NW": NW, "ETA": ETA,
                      "note": "helicity-spin locking: beta_up(sp+1)=beta_dn(sp-1)"},
           "rows": rows, "store": store}
    with open("data/cpge_helicity.json", "w") as f:
        json.dump(out, f, indent=2)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for iA, A0 in enumerate((0.06, 0.12)):
        for iS, sp in enumerate((+1.0, -1.0)):
            ax = axes[iA, iS]
            key = f"A0_{A0:.2f}_sp{sp:+.0f}"
            dat = store[key]
            om = np.array(dat["omega"])
            ax.plot(om, np.array(dat["beta_up"]) * 1e3, "r-", lw=1.2, label=r"$\uparrow$")
            ax.plot(om, np.array(dat["beta_dn"]) * 1e3, "b-", lw=1.2, label=r"$\downarrow$")
            ax.plot(om, np.array(dat["beta_tot"]) * 1e3, "k--", lw=1.5, label="total")
            ax2 = ax.twinx()
            ax2.plot(om, np.array(dat["P"]), "g:", lw=1.0, alpha=0.8)
            ax2.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="gray", ls=":", lw=0.6)
            ax.set_xlim(0, 60)
            if iA == 1:
                ax.set_xlabel(r"$\hbar\omega$ (meV)")
            if iS == 0:
                ax.set_ylabel(r"$\beta^{xy}\,(\times10^{-3})$")
            ax.set_title(rf"$A_0={A0:.2f}$, sp={sp:+.0f}")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(rf"Paper H -- helicity dependence of spin-resolved CPGE, $M={M0:+.0f}$ meV, $J_0={J0:.0f}$ meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("figs/cpge_helicity_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/cpge_helicity.json")
    print("Saved: figs/cpge_helicity_panel.png")


if __name__ == "__main__":
    main()

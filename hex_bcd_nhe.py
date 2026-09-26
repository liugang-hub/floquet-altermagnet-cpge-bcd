# -*- coding: utf-8 -*-
"""
Paper H -- Script 10 (v2): Berry-curvature dipole (BCD) / nonlinear Hall
effect (NHE) on the hexagonal lattice.  Continuous k.p + TRUE hexagonal 1st
BZ (shared module hex_model, v2.1 with the corrected J0 derivatives).

Symmetry logic (the upgrade from the square lattice):
  * BCD is a Fermi-surface property: on ANY lattice, a fully occupied band
    gives D_a = int d2k d_a Omega = 0 by periodicity (topological identity).
    EF must cut a Fermi surface -- here, a hole pocket around Gamma inside
    the mass-inversion ring (the valence top is a Mexican hat for B < 0).
  * The d-wave altermagnet fd = (3/8)(ky^2-kx^2) (MnTe Cm'c'm orientation)
    is EVEN under BOTH cartesian mirrors kx->-kx, ky->-ky; Omega is even in
    kx and ky separately, so with tilt T = 0 the residual symmetry is C2v
    and the BCD vanishes:  D_x = D_y = 0  (checked numerically as control).
  * A tilt T kx in h0 breaks ONLY the kx mirror (My is preserved), so
    D_y == 0 exactly and D_x opens:  the transverse NHE current
    J_y ~ D_x E_x^2.  [v1 claimed "tilt opens D_y" -- that is wrong for
    this fd orientation; the algebra and numerics below show D_x.]
  * Altermagnet spin splitting (J0 term, now entering Omega via the FIXED
    derivatives) makes D_x^up != D_x^dn -> spin nonlinear Hall channel
    D_x^spin = D_x^up - D_x^dn.
  * Floquet drive A0 renormalizes At_s, Mt_s -> tunes pocket shape and the
    dipole continuously (A0 scan); EF scan at fixed A0 shows the Fermi-
    surface character; T = 0 vs T = 30 meV shows thermal robustness.

Outputs:
  - data/hex_bcd_nhe.json
  - figs/hex_bcd_nhe_panel.png
  - terminal summary
"""
import os, json, time
import numpy as np
import hex_model as hm

# ------------------------------ parameters ---------------------------------
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
HW = hm.HW_PARAM
SP = 1.0
J0 = float(os.environ.get("PH_J0", "160.0"))
TILT = float(os.environ.get("PH_TILT", "30.0"))
NK = int(os.environ.get("PH_NK", "400"))
# EF scan: depth (meV) below the valence-band top, and (T, tilt) control rows
DELTAS = [float(x) for x in os.environ.get("PH_DELTAS", "0.5,1,2,3,4,6,8").split(",")]
CTRL = [(0.0, 0.0), (0.0, 30.0), (30.0, 30.0)]     # (T_meV, TILT)
A0_EF = 0.06
DELTA_METAL = float(os.environ.get("PH_DELTA_METAL", "3.0"))
A0_LIST = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]


def fermi_step(E, EF, T):
    """Occupancy factor for the BCD: step at T=0, Fermi function at T>0."""
    if T <= 1e-9:
        return (E < EF).astype(float)
    return 1.0 / (1.0 + np.exp((E - EF) / T))


def band_top(kx, ky, mask, spin, At, Mt, A0, T):
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, T, J0)
    return float(Ev[mask].max())


def bcd_vec(kx, ky, mask, d2k, spin, At, Mt, A0, TILT_H, T_meV, EF):
    """Vectorized BCD: D_a = sum f_T(E_v) d_a Omega * mask * d2k.

    TILT_H: mirror-breaking tilt in h0 (meV); T_meV: temperature (k_B T) for
    the Fermi occupancy.  Kept separate: the tilt is a Hamiltonian parameter,
    the temperature only enters the occupation factor.
    """
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    _, dOx, dOy = hm.omega_full(kx, ky, spin, At, Mt, A0, TILT_H, J0)
    f = fermi_step(Ev, EF, T_meV)
    occ = float((f * mask).sum()) / float(mask.sum())
    Dx = float(np.sum(f * dOx * mask) * d2k)
    Dy = float(np.sum(f * dOy * mask) * d2k)
    return Dx, Dy, occ


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    kx, ky, mask, d2k = hm.hex_mesh(NK)
    out = {"params": {}, "ef_scan": [], "a0_scan": [], "topology_check": {}}
    out["params"] = {
        "A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP,
        "J0": J0, "NK": NK, "A0_ef": A0_EF, "delta_metal": DELTA_METAL,
        "deltas": DELTAS, "ctrl_rows": CTRL, "a0_list": A0_LIST,
        "lattice": "triangular, hexagonal 1st BZ, R = 4pi/3",
        "model": "continuous k.p + hex BZ (v2.1); fd=(3/8)(ky^2-kx^2); "
                 "BCD D_a = int f(E) d_a Omega; E_v = h0 - |d|, h0 has tilt T kx",
    }

    # ---- topology identity: fully occupied band -> D == 0 ----------------
    At0, Mt0 = hm.floquet_params(A, B, D, M0, HW, A0_EF)
    _, dOx0, dOy0 = hm.omega_full(kx, ky, "up", At0, Mt0, A0_EF, 30.0, J0)
    out["topology_check"] = {
        "Dx_all_occupied": float(np.sum(dOx0 * mask) * d2k),
        "Dy_all_occupied": float(np.sum(dOy0 * mask) * d2k),
    }
    print("[topology identity] fully occupied valence band:")
    print(f"  Dx = {out['topology_check']['Dx_all_occupied']:+.3e}  "
          f"Dy = {out['topology_check']['Dy_all_occupied']:+.3e}   (must be ~0)")

    # ---- (a) EF scan: 3 control rows (T, tilt) ---------------------------
    print("\n== hexagonal BCD: EF scan, A0 = 0.06 ==")
    At_ef, Mt_ef = hm.floquet_params(A, B, D, M0, HW, A0_EF)
    top = {s: band_top(kx, ky, mask, s, At_ef, Mt_ef, A0_EF, TILT) for s in ("up", "dn")}
    top_max = max(top.values())
    print(f"  valence-band top at A0=0.06: up={top['up']:+.3f}  "
          f"dn={top['dn']:+.3f}  max={top_max:+.3f} meV (spin split = altermagnet)")
    for (Ttmp, Ttilt) in CTRL:
        topT = {s: band_top(kx, ky, mask, s, At_ef, Mt_ef, A0_EF, Ttilt)
                for s in ("up", "dn")}
        topT_max = max(topT.values())
        print(f"  --- T={Ttmp:.0f} meV, tilt={Ttilt:.0f} meV "
              f"(top: up={topT['up']:+.2f}, dn={topT['dn']:+.2f}) ---")
        for delta in DELTAS:
            EF = topT_max - delta
            r = {"T": Ttmp, "TILT": Ttilt, "delta": delta, "EF": EF,
                 "top_max": topT_max}
            for spin in ("up", "dn"):
                Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, spin, At_ef, Mt_ef,
                                      A0_EF, Ttilt, Ttmp, EF)
                r[f"Dx_{spin}"] = Dx
                r[f"Dy_{spin}"] = Dy
                r[f"occ_{spin}"] = occ
            r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
            r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
            r["Dx_spin"] = r["Dx_up"] - r["Dx_dn"]     # spin BCD channel
            out["ef_scan"].append(r)
            if delta in (1.0, 3.0, 6.0) or abs(EF - topT_max + 1.0) < 1e-9:
                print(f"    delta={delta:4.1f} EF={EF:+6.2f}: "
                      f"Dx(up,dn,tot,spin)=({r['Dx_up']:+.3e},{r['Dx_dn']:+.3e},"
                      f"{r['Dx_tot']:+.3e},{r['Dx_spin']:+.3e})  "
                      f"Dy_tot={r['Dy_tot']:+.2e}  occ={r['occ_up']:.3f}")

    # ---- (b) A0 scan at metallic EF (room T = 30 meV, tilt 30) -----------
    print(f"\n== hexagonal BCD: A0 scan, T = 30 meV, tilt = 30 meV, "
          f"EF = top_max - {DELTA_METAL:.1f} meV (at A0=0.06) ==")
    top_metal = {s: band_top(kx, ky, mask, s, At_ef, Mt_ef, A0_EF, TILT)
                 for s in ("up", "dn")}
    EF_metal = max(top_metal.values()) - DELTA_METAL
    print(f"  EF_metal = {EF_metal:+.3f} meV")
    for A0 in A0_LIST:
        At, Mt = hm.floquet_params(A, B, D, M0, HW, A0)
        r = {"A0": A0, "EF": EF_metal, "Mt_up": Mt["up"], "Mt_dn": Mt["dn"],
             "At_up": At["up"], "At_dn": At["dn"]}
        for spin in ("up", "dn"):
            Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, spin, At, Mt, A0, TILT,
                                  30.0, EF_metal)
            r[f"Dx_{spin}"] = Dx
            r[f"Dy_{spin}"] = Dy
            r[f"occ_{spin}"] = occ
        r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
        r["Dy_tot"] = r["Dy_up"] + r["Dy_dn"]
        r["Dx_spin"] = r["Dx_up"] - r["Dx_dn"]
        out["a0_scan"].append(r)
        print(f"  A0={A0:.2f} Mt(up,dn)=({Mt['up']:+6.2f},{Mt['dn']:+6.2f}): "
              f"Dx(up,dn,tot,spin)=({r['Dx_up']:+.3e},{r['Dx_dn']:+.3e},"
              f"{r['Dx_tot']:+.3e},{r['Dx_spin']:+.3e})  "
              f"Dy_tot={r['Dy_tot']:+.2e}  occ(up,dn)=({r['occ_up']:.3f},{r['occ_dn']:.3f})")

    with open("data/hex_bcd_nhe.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["ef_scan"]
    fig = plt.figure(figsize=(12.5, 3.6))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    # panel 1: Dx_tot vs delta for the three control rows
    for (Ttmp, Ttilt) in CTRL:
        sub = [r for r in rows if r["T"] == Ttmp and r["TILT"] == Ttilt]
        lbl = rf"$T={Ttmp:.0f}$, tilt={Ttilt:.0f}"
        ax1.plot([r["delta"] for r in sub], [r["Dx_tot"] for r in sub],
                 "o-", lw=1.3, ms=3.5, label=lbl)
    ax1.axhline(0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel(r"$\delta=E_{\rm top}-E_F$ (meV)")
    ax1.set_ylabel(r"$D_x^{\rm tot}$")
    ax1.set_title(rf"BCD vs depth, $A_0={A0_EF:.2f}$")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # panel 2: spin-resolved Dx at T=30, tilt=30
    sub = [r for r in rows if r["T"] == 30.0 and r["TILT"] == 30.0]
    ax2.plot([r["delta"] for r in sub], [r["Dx_up"] for r in sub], "ro-",
             lw=1.3, ms=3.5, label=r"$D_x^{\uparrow}$")
    ax2.plot([r["delta"] for r in sub], [r["Dx_dn"] for r in sub], "bo-",
             lw=1.3, ms=3.5, label=r"$D_x^{\downarrow}$")
    ax2.plot([r["delta"] for r in sub], [r["Dx_spin"] for r in sub], "g^-",
             lw=1.3, ms=3.5, label=r"$D_x^{s}$")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel(r"$\delta$ (meV)")
    ax2.set_ylabel(r"$D_x$")
    ax2.set_title(r"Spin-resolved BCD, $T$=30 meV")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    # panel 3: A0 scan
    a0s = out["a0_scan"]
    ax3.plot([r["A0"] for r in a0s], [r["Dx_up"] for r in a0s], "ro-",
             lw=1.3, ms=3.5, label=r"$D_x^{\uparrow}$")
    ax3.plot([r["A0"] for r in a0s], [r["Dx_dn"] for r in a0s], "bo-",
             lw=1.3, ms=3.5, label=r"$D_x^{\downarrow}$")
    ax3.plot([r["A0"] for r in a0s], [r["Dx_tot"] for r in a0s], "k--s",
             lw=1.2, ms=3, label=r"$D_x^{\rm tot}$")
    ax3.plot([r["A0"] for r in a0s], [r["Dx_spin"] for r in a0s], "g^-",
             lw=1.3, ms=3.5, label=r"$D_x^{s}$")
    ax3.axhline(0, color="gray", ls=":", lw=0.8)
    ax3.set_xlabel(r"$A_0$ (nm$^{-1}$)")
    ax3.set_ylabel(r"$D_x$")
    ax3.set_title(rf"Floquet tuning, $E_F={EF_metal:+.0f}$ meV")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # panel 4: occupancy maps (up, dn) at the A0-scan working point
    At_w, Mt_w = hm.floquet_params(A, B, D, M0, HW, A0_EF)
    occ_up = np.zeros_like(kx)
    occ_dn = np.zeros_like(kx)
    for spin, occ in (("up", occ_up), ("dn", occ_dn)):
        Ev = hm.valence_energy(kx, ky, spin, At_w, Mt_w, A0_EF, TILT, J0)
        occ[:] = (Ev < EF_metal).astype(float)
    import matplotlib.colors as mcolors
    occ_comb = np.where(mask, occ_up + 2.0 * occ_dn, np.nan)
    im = ax4.imshow(occ_comb.T, origin="lower",
                    extent=[-hm.R_HEX, hm.R_HEX, -hm.R_HEX, hm.R_HEX],
                    cmap=mcolors.ListedColormap(["white", "red", "blue", "purple"]),
                    vmin=0, vmax=3, aspect="equal")
    ax4.set_xlabel(r"$k_x$ (nm$^{-1}$)")
    ax4.set_ylabel(r"$k_y$")
    ax4.set_title(rf"occ: red=$\uparrow$, blue=$\downarrow$, "
                  rf"purple=both, $E_F={EF_metal:+.1f}$")
    import matplotlib.patches as mpatches
    ax4.legend(handles=[mpatches.Patch(color="red", label=r"$\uparrow$ only"),
                        mpatches.Patch(color="blue", label=r"$\downarrow$ only"),
                        mpatches.Patch(color="purple", label="both")],
               fontsize=6, loc="lower left")
    for n in range(6):
        th = n * np.pi / 3.0 + np.pi / 6.0
        ax4.plot([0, hm.R_HEX * np.cos(th)], [0, hm.R_HEX * np.sin(th)],
                 color="gray", lw=0.5, alpha=0.6)

    fig.suptitle(rf"hexagonal-lattice BCD/NHE: $J_0={J0:.0f}$ meV, "
                 rf"$M={M0:+.0f}$ meV, $A={A:.0f}$ meV", fontsize=12, y=0.965)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("figs/hex_bcd_nhe_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/hex_bcd_nhe.json")
    print("Saved: figs/hex_bcd_nhe_panel.png")


if __name__ == "__main__":
    main()

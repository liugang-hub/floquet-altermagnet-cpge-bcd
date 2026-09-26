# -*- coding: utf-8 -*-
"""
Paper H -- Script 11: dense EF scan of the spin-resolved BCD (Dx_spin curve).

Purpose: continuous Dx_spin(E_F) curve to be merged into Fig.3b.  The key
physics window is the altermagnet single-spin window: at A0 = 0.06 the band
tops are up = +3.785 / dn = +0.385 meV (J0 = 160, tilt = 30), i.e. a
spin-split window of ~3.4 meV in which only the up-spin band has a Fermi
surface at T = 0  ->  Dx_spin = Dx_up (100 % spin-polarized BCD).

Scans:
  (a) T = 0   (k_B T -> 0): step occupancy, single-spin window resolution
  (b) T = 30 meV (~348 K): room-temperature charge NHE / spin channel decay

EF grid: delta = top_max - EF from 0.0 to PH_EF_MAX (default 8.0) meV in
steps of PH_EF_STEP (default 0.1 meV).  Fixed A0 = PH_A0 (default 0.06),
J0 = PH_J0 (default 160), tilt = PH_TILT (default 30).

Outputs:
  - data/hex_ef_dense.json  (all curves + extracted features)
  - figs/hex_ef_dense.png   (3 panels: Dx_spin(delta), spin-resolved Dx(delta),
                             occ_up/occ_dn(delta) marking the single-spin window)

Usage (formal, run by user):
  set PH_NK=400
  python scan_ef_dense.py
"""
import os, json, time
import numpy as np
import hex_model as hm

# ------------------------------ parameters ---------------------------------
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
HW = hm.HW_PARAM
J0 = float(os.environ.get("PH_J0", "160.0"))
TILT = float(os.environ.get("PH_TILT", "30.0"))
NK = int(os.environ.get("PH_NK", "400"))
A0 = float(os.environ.get("PH_A0", "0.06"))
EF_MAX = float(os.environ.get("PH_EF_MAX", "8.0"))
EF_STEP = float(os.environ.get("PH_EF_STEP", "0.1"))
TS = [0.0, 30.0]                       # temperatures for the two curves


def fermi_step(E, EF, T):
    if T <= 1e-9:
        return (E < EF).astype(float)
    return 1.0 / (1.0 + np.exp((E - EF) / T))


def bcd_vec(kx, ky, mask, d2k, spin, At, Mt, A0h, TILT_H, T_meV, EF):
    """Vectorized BCD, self-contained copy (see hex_bcd_nhe.bcd_vec)."""
    Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0h, TILT_H, J0)
    _, dOx, dOy = hm.omega_full(kx, ky, spin, At, Mt, A0h, TILT_H, J0)
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
    At, Mt = hm.floquet_params(A, B, D, M0, HW, A0)

    # band tops and the altermagnet window
    top = {}
    for spin in ("up", "dn"):
        Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT, J0)
        top[spin] = float(Ev[mask].max())
    top_max = max(top.values())
    delta_win = top["up"] - top["dn"]
    print(f"[A0={A0:.2f}, J0={J0:.0f}, tilt={TILT:.0f}, NK={NK}]")
    print(f"  band tops: up={top['up']:+.3f}  dn={top['dn']:+.3f}  "
          f"max={top_max:+.3f} meV  ->  single-spin window = {delta_win:+.3f} meV")

    deltas = np.arange(0.0, EF_MAX + 1e-9, EF_STEP)
    out = {"params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW,
                      "J0": J0, "NK": NK, "A0": A0, "TILT": TILT,
                      "EF_MAX": EF_MAX, "EF_STEP": EF_STEP,
                      "top_up": top["up"], "top_dn": top["dn"],
                      "window_meV": delta_win},
           "curves": {}}

    for Tk in TS:
        rows = []
        for delta in deltas:
            EF = top_max - delta
            r = {"delta": float(delta), "EF": float(EF)}
            for spin in ("up", "dn"):
                Dx, Dy, occ = bcd_vec(kx, ky, mask, d2k, spin, At, Mt,
                                      A0, TILT, Tk, EF)
                r[f"Dx_{spin}"] = Dx
                r[f"occ_{spin}"] = occ
            r["Dx_tot"] = r["Dx_up"] + r["Dx_dn"]
            r["Dx_spin"] = r["Dx_up"] - r["Dx_dn"]
            rows.append(r)
        out["curves"][f"T{Tk:g}"] = rows

        # ---- feature extraction ------------------------------------------
        dxs = np.array([r["Dx_spin"] for r in rows])
        occ_dn = np.array([r["occ_dn"] for r in rows])
        # single-spin window: T=0 and dn fully occupied (no dn Fermi surface)
        win = np.where(np.abs(occ_dn - 1.0) < 1e-9)[0]
        if Tk < 1e-9 and len(win):
            d_win = float(deltas[win].max())
            print(f"  T={Tk:g}: single-spin window (occ_dn=1.000) delta <= "
                  f"{d_win:.2f} meV;  |Dx_spin|max in window = "
                  f"{np.max(np.abs(dxs[win])):.3f} at delta="
                  f"{deltas[win][np.argmax(np.abs(dxs[win]))]:.2f}")
        # zero crossings of Dx_spin
        zc = []
        for i in range(1, len(rows)):
            if dxs[i-1] * dxs[i] < 0:
                zc.append(float(0.5 * (deltas[i-1] + deltas[i])))
        if zc:
            print(f"  T={Tk:g}: Dx_spin zero crossings at delta = "
                  + ", ".join(f"{z:.2f}" for z in zc))
        print(f"  T={Tk:g}: Dx_tot range [{rows[0]['Dx_tot']:+.3f}, "
              f"{rows[-1]['Dx_tot']:+.3f}],  "
              f"Dx_spin range [{dxs.min():+.3f}, {dxs.max():+.3f}]")

    with open("data/hex_ef_dense.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for Tk in TS:
        rows = out["curves"][f"T{Tk:g}"]
        ds = [r["delta"] for r in rows]
        lbl = rf"$T={Tk:g}$ meV"
        axes[0].plot(ds, [r["Dx_spin"] for r in rows], "o-", lw=1.2, ms=2,
                     label=lbl)
        axes[1].plot(ds, [r["Dx_up"] for r in rows], "r.-", lw=1.0, ms=2,
                     label=lbl + r", $\uparrow$")
        axes[1].plot(ds, [r["Dx_dn"] for r in rows], "b.-", lw=1.0, ms=2,
                     label=lbl + r", $\downarrow$")
        axes[2].plot(ds, [r["occ_up"] for r in rows], "r.--", lw=1.0, ms=2,
                     label=lbl + r", occ$_\uparrow$")
        axes[2].plot(ds, [r["occ_dn"] for r in rows], "b.--", lw=1.0, ms=2,
                     label=lbl + r", occ$_\downarrow$")
    axes[0].axhline(0, color="gray", ls=":", lw=0.8)
    axes[0].axvspan(0, delta_win, color="gold", alpha=0.18,
                    label="single-spin window")
    axes[0].set_xlabel(r"$\delta=E_{\rm top}-E_F$ (meV)")
    axes[0].set_ylabel(r"$D_x^{\rm spin}$")
    axes[0].set_title(rf"$D_x^s(\delta)$, $A_0={A0:.2f}$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="gray", ls=":", lw=0.8)
    axes[1].axvspan(0, delta_win, color="gold", alpha=0.18)
    axes[1].set_xlabel(r"$\delta$ (meV)")
    axes[1].set_ylabel(r"$D_x$")
    axes[1].set_title("Spin-resolved BCD")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    axes[2].axhline(1.0, color="gray", ls=":", lw=0.8)
    axes[2].axvspan(0, delta_win, color="gold", alpha=0.18)
    axes[2].set_xlabel(r"$\delta$ (meV)")
    axes[2].set_ylabel("occupancy")
    axes[2].set_title("Fermi-surface character")
    axes[2].legend(fontsize=6)
    axes[2].grid(alpha=0.3)
    fig.suptitle(rf"Paper H -- dense EF scan of spin BCD, $J_0={J0:.0f}$ meV, "
                 rf"tilt={TILT:.0f}$\,$meV, window={delta_win:.2f}$\,$meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("figs/hex_ef_dense.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/hex_ef_dense.json")
    print("Saved: figs/hex_ef_dense.png")


if __name__ == "__main__":
    main()

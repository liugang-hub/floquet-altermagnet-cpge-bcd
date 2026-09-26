"""A0 scan of BCD at a *fixed* Fermi energy (Fig.3c variant).

Default EF=+1.0 meV matches the caption in fig3_bcd_nhe.png.
Use this if you want to keep "EF=+1 meV" in the figure; otherwise,
the existing data corresponds to EF=+0.785 meV (top_max - 3.0 meV).

Usage (kwant2 terminal):
    set PH_EF=1.0
    python scan_a0_at_ef.py
Optional: set PH_REPLOT=1 to skip the scan and only re-render the
figure from the existing data/hex_a0_at_ef.json (data file untouched).
Output:
    data/hex_a0_at_ef.json
    figs/hex_a0_at_ef.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hex_model as hm

A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
J0 = float(os.environ.get("PH_J0", "160.0"))
TILT = float(os.environ.get("PH_TILT", "30.0"))
EF = float(os.environ.get("PH_EF", "1.0"))   # fixed Fermi energy in meV
NK = int(os.environ.get("PH_NK", "400"))

kx, ky, mask, d2k = hm.hex_mesh(NK)
A0_list = np.linspace(0.0, 0.12, 25)

def bcd_at_ef(A0):
    At, Mt = hm.floquet_params(A, B, D, M0, hm.HW_PARAM, A0)
    rows = []
    for spin in ("up", "dn"):
        Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT, J0)
        _, dOx, _ = hm.omega_full(kx, ky, spin, At, Mt, A0, TILT, J0)
        occ = np.where(Ev > EF, 1.0, 0.0)   # T=0 step
        Dx = float(np.sum(occ * dOx * mask) * d2k)
        rows.append({"spin": spin, "A0": float(A0), "Mt": float(Mt[spin]), "Dx": Dx,
                     "occ": float(np.sum(occ * mask) / np.sum(mask))})
    return rows

if os.environ.get("PH_REPLOT", "0") == "1":
    # Reuse the NK=400 data already on disk; only re-render the figure.
    # Does NOT overwrite data/hex_a0_at_ef.json.
    with open("data/hex_a0_at_ef.json") as f:
        out = json.load(f)
    results = out["results"]
    print(f"PH_REPLOT=1: reused data/hex_a0_at_ef.json "
          f"(EF={out['params']['EF']}, NK={out['params']['NK']})")
else:
    results = [bcd_at_ef(a0) for a0 in A0_list]

    # Save JSON
    out = {"params": {"J0": J0, "TILT": TILT, "EF": EF, "NK": NK},
           "results": results}
    with open("data/hex_a0_at_ef.json", "w") as f:
        json.dump(out, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(4.0, 3.2))
A0_arr = np.array([r[0]["A0"] for r in results])
Dx_up = np.array([r[0]["Dx"] for r in results])
Dx_dn = np.array([r[1]["Dx"] for r in results])
Dx_tot = Dx_up + Dx_dn
Dx_spin = Dx_up - Dx_dn

ax.plot(A0_arr, Dx_up, "ro-", lw=1.2, ms=3, label=r"$D_x^{\uparrow}$")
ax.plot(A0_arr, Dx_dn, "bo-", lw=1.2, ms=3, label=r"$D_x^{\downarrow}$")
ax.plot(A0_arr, Dx_tot, "ks--", lw=1.2, ms=3, label=r"$D_x^{\rm tot}$")
ax.plot(A0_arr, Dx_spin, "g^-", lw=1.2, ms=3, label=r"$D_x^{\rm spin}$")
ax.axvline(0.12, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"$A_0$ (nm$^{-1}$)")
ax.set_ylabel(r"$D_x$ (nm)")
ax.set_title(f"Floquet tuning at fixed $E_F={EF:+.1f}$ meV (tilt={TILT} meV·nm)")
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("figs/hex_a0_at_ef.png", dpi=300, bbox_inches="tight")
print(f"Saved data/hex_a0_at_ef.json and figs/hex_a0_at_ef.png (EF={EF}, NK={NK})")

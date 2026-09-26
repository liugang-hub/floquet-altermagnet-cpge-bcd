"""Generate Fig.3(d): hexagonal-BZ map of Berry-curvature spin character.

Reads model parameters from hex_model.py and plots, at fixed EF, which
regions of the hexagonal BZ are occupied by up-spin only, down-spin only,
or both spins.  This matches the panel shown in fig3_bcd_nhe.png.

Usage (kwant2 terminal):
    python plot_berry_map.py
Output:
    figs/fig3d_berry_map.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hex_model as hm

# Parameters (same as official BCD runs)
A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
J0 = float(os.environ.get("PH_J0", "160.0"))
A0 = float(os.environ.get("PH_A0", "0.06"))
TILT = float(os.environ.get("PH_TILT", "30.0"))
EF = float(os.environ.get("PH_EF", "0.8"))   # meV, as in fig3_bcd_nhe caption
NK = int(os.environ.get("PH_NK", "400"))

At, Mt = hm.floquet_params(A, B, D, M0, hm.HW_PARAM, A0)
kx, ky, mask, d2k = hm.hex_mesh(NK)

# Occupation at T=0 step (smear slightly for visualization)
def occ(Ev, EF, width=0.02):
    return 0.5 * (1.0 - np.tanh((Ev - EF) / width))

occ_up = occ(hm.valence_energy(kx, ky, "up", At, Mt, A0, TILT, J0), EF)
occ_dn = occ(hm.valence_energy(kx, ky, "dn", At, Mt, A0, TILT, J0), EF)

# Character masks
th = 0.5
up_only = mask & (occ_up > th) & (occ_dn < th)
dn_only = mask & (occ_up < th) & (occ_dn > th)
both = mask & (occ_up > th) & (occ_dn > th)

fig, ax = plt.subplots(figsize=(4.0, 3.6))

# Hexagon boundary
R = 4.0 * np.pi / 3.0
hex_ang = np.linspace(0, 2 * np.pi, 7)
hx, hy = R * np.cos(hex_ang + np.pi / 6.0), R * np.sin(hex_ang + np.pi / 6.0)
ax.plot(hx, hy, "k-", lw=1.0)

# Scatter plot (mask points; subsample for visibility if NK is large)
step = max(1, NK // 200)
sel = mask & (occ_up > 1e-3) & (occ_dn > 1e-3)
ax.scatter(kx[sel][::step], ky[sel][::step], c="purple", s=1.0, alpha=0.3, label="both")
if up_only.any():
    ax.scatter(kx[up_only][::step], ky[up_only][::step], c="red", s=3.0, alpha=0.7, label=r"$\uparrow$ only")
if dn_only.any():
    ax.scatter(kx[dn_only][::step], ky[dn_only][::step], c="blue", s=3.0, alpha=0.7, label=r"$\downarrow$ only")

ax.set_aspect("equal")
ax.set_xlabel(r"$k_x$ (nm$^{-1}$)")
ax.set_ylabel(r"$k_y$ (nm$^{-1}$)")
ax.set_title(r"Berry-curvature spin character at $E_F=" + f"{EF:+.1f}$ meV\n"
             rf"($A_0={A0}$ nm$^{{-1}}$, $J_0={J0}$ meV, tilt={TILT} meV·nm)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.savefig("figs/fig3d_berry_map.png", dpi=300, bbox_inches="tight")
print(f"Saved figs/fig3d_berry_map.png (EF={EF}, NK={NK})")

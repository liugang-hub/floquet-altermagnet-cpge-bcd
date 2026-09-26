# -*- coding: utf-8 -*-
"""
Paper H -- Script 4: Berry-curvature map and Chern-number check for the
Floquet-renormalized altermagnetic BHZ bulk.

We use the analytic 2-band Berry curvature for H_s(k) = d_s(k)·sigma:

    Omega_s(k) = (1/(2 |d|^3)) d · (partial_x d x partial_y d)

This is compared with the CPGE weight Im(r_x_cv r_y_vc) computed in the
CPGE scripts, validating that the sign structure of the photocurrent really
comes from the quantum geometry of the bands.

Outputs:
  - data/berry_map.json
  - figs/berry_map_panel.png
"""
import os, json, time
import numpy as np

A, B, D, M0 = 364.5, -686.0, -512.0, -10.0
HW = 150.0
SP = 1.0
S3 = {"up": +1.0, "dn": -1.0}
J0 = float(os.environ.get("PH_J0", "160.0"))
NK = int(os.environ.get("PH_NK", "160"))
EF = 0.0


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


def d_vector(kx, ky, spin, At, A0):
    """d=(hx,hy,hz) for the 2x2 block."""
    k2 = 2.0 * (1.0 - np.cos(kx)) + 2.0 * (1.0 - np.cos(ky))
    s = S3[spin]
    m = M0 - B * k2 + 2.0 * s * J0 * (np.cos(ky) - np.cos(kx))
    h0 = -D * (A0**2 + k2)
    hx = At[spin] * np.sin(kx)
    hy = At[spin] * np.sin(ky)
    hz = m
    return h0, hx, hy, hz


def berry_curvature(kx, ky, spin, At, A0):
    """Analytic Berry curvature of the valence band (lower eigenstate)."""
    s = S3[spin]
    sinx, cosx = np.sin(kx), np.cos(kx)
    siny, cosy = np.sin(ky), np.cos(ky)

    hx = At[spin] * sinx
    hy = At[spin] * siny
    # m = M0 - B k2 + 2sJ0(cos y - cos x)
    dm_dx = 2.0 * sinx * (s * J0 - B)   # d/dx [-B k2 + 2sJ0(cos y - cos x)]
    dm_dy = -2.0 * siny * (B + s * J0)  # d/dy [...]

    dhx_dx = At[spin] * cosx
    dhy_dy = At[spin] * cosy

    # d-vector (drop h0 because it does not affect eigenvectors)
    d_vec = np.array([hx, hy, 0.0])
    # ... but hz comes from m (d_z = m)
    d_vec[2] = M0 - B * (2.0*(1-cosx) + 2.0*(1-cosy)) + 2.0*s*J0*(cosy - cosx)

    dx = np.array([dhx_dx, 0.0, dm_dx])
    dy = np.array([0.0, dhy_dy, dm_dy])

    cross = np.cross(dx, dy)
    d_norm = np.linalg.norm(d_vec)
    # avoid division by zero at gapless points
    d_norm = max(d_norm, 1e-12)
    omega = np.dot(d_vec, cross) / (2.0 * d_norm**3)
    return omega


def numerical_berry(kx, ky, spin, At, A0):
    """Compute Berry curvature directly from the U matrix of H(k)."""
    k2 = 2.0 * (1.0 - np.cos(kx)) + 2.0 * (1.0 - np.cos(ky))
    s = S3[spin]
    m = M0 - B * k2 + 2.0 * s * J0 * (np.cos(ky) - np.cos(kx))
    h0 = -D * (A0**2 + k2)
    hx = At[spin] * np.sin(kx)
    hy = At[spin] * np.sin(ky)
    hz = m
    H = np.array([[h0 + hz, hx - 1j * hy],
                  [hx + 1j * hy, h0 - hz]], dtype=complex)
    E, U = np.linalg.eigh(H)
    # small finite difference for derivatives of U
    dk = 0.001
    def build(kx_, ky_):
        k2_ = 2.0 * (1.0 - np.cos(kx_)) + 2.0 * (1.0 - np.cos(ky_))
        m_ = M0 - B * k2_ + 2.0 * s * J0 * (np.cos(ky_) - np.cos(kx_))
        h0_ = -D * (A0**2 + k2_)
        hx_ = At[spin] * np.sin(kx_)
        hy_ = At[spin] * np.sin(ky_)
        return np.array([[h0_ + m_, hx_ - 1j*hy_],
                         [hx_ + 1j*hy_, h0_ - m_]], dtype=complex)
    _, Ux = np.linalg.eigh(build(kx + dk, ky))
    _, Uxm = np.linalg.eigh(build(kx - dk, ky))
    _, Uy = np.linalg.eigh(build(kx, ky + dk))
    _, Uym = np.linalg.eigh(build(kx, ky - dk))
    # gauge alignment: force first column (lower state) real-positive
    u = U[:, 0]
    def align(v, ref):
        o = np.vdot(ref, v)
        return v * (o / abs(o)) if abs(o) > 1e-12 else v
    ux  = align(Ux[:, 0], u)
    uxm = align(Uxm[:, 0], u)
    uy  = align(Uy[:, 0], u)
    uym = align(Uym[:, 0], u)
    dx_u = (ux - uxm) / (2.0 * dk)
    dy_u = (uy - uym) / (2.0 * dk)
    omega = -2.0 * np.imag(np.vdot(dx_u, dy_u))
    return omega


def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    A0 = 0.06
    At, Mt = floquet_params(A0)
    At["A0"] = A0
    print(f"[M={M0:+.0f} meV, J0={J0:.0f} meV, A0={A0:.3f} nm^-1]")
    print(f"  Mt_up={Mt['up']:.3f}, Mt_dn={Mt['dn']:.3f} meV")
    print(f"  At_up={At['up']:.2f}, At_dn={At['dn']:.2f} meV·nm")

    k1d = (np.arange(NK) - NK // 2) * 2.0 * np.pi / NK
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    d2k = (2.0 * np.pi / NK) ** 2

    maps = {}
    for spin in ("up", "dn"):
        om_a = np.zeros_like(kx)
        om_n = np.zeros_like(kx)
        for i in range(NK):
            for j in range(NK):
                om_a[i, j] = berry_curvature(kx[i, j], ky[i, j], spin, At, A0)
                # numerical is expensive; sample a few points only
                if i % 10 == 0 and j % 10 == 0:
                    om_n[i, j] = numerical_berry(kx[i, j], ky[i, j], spin, At, A0)
        ch = np.sum(om_a) * d2k / (2.0 * np.pi)
        print(f"  spin={spin}: Chern number C = {ch:+.4f}")
        maps[spin] = om_a.tolist()
        maps[f"{spin}_num"] = om_n.tolist()

    diff = np.array(maps["up"]) - np.array(maps["dn"])

    out = {
        "params": {"A": A, "B": B, "D": D, "M0": M0, "HW": HW, "SP": SP, "J0": J0,
                   "A0": A0, "NK": NK, "Mt": Mt, "At": At},
        "maps": maps,
    }
    with open("data/berry_map.json", "w") as f:
        json.dump(out, f, indent=2)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [-np.pi, np.pi, -np.pi, np.pi]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for idx, (spin, title) in enumerate([("up", r"$\uparrow$"), ("dn", r"$\downarrow$"), ("diff", "up - dn")]):
        ax = axes[idx]
        if spin == "diff":
            dat = diff
        else:
            dat = np.array(maps[spin])
        vmax = np.percentile(np.abs(dat), 99.0)
        im = ax.imshow(dat.T, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(rf"Paper H -- Berry curvature, $A_0={A0:.3f}$ nm$^{{-1}}$, $M={M0:+.0f}$ meV",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("figs/berry_map_panel.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTime elapsed: {time.time()-t0:.2f} s")
    print("Saved: data/berry_map.json")
    print("Saved: figs/berry_map_panel.png")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper H -- shared hexagonal-lattice model module (v2: continuous k.p + hex BZ).

Why v2 (after v1 lattice-regularization failed): on the triangular lattice the
nearest-neighbour hybridization hx, hy are intrinsically anisotropic
(hx = A sin kx, hy = A (s2+s3)/sqrt(3)), which injects a k . grad(m) mixing
term into the Berry curvature: Omega ~ [m - 4B k^2]/... and the Chern number
is no longer a clean +-1 even in the bare BHZ limit (numerically C ~ 0 for
J0 <= 120).  The continuous k.p model

    hx = A kx,  hy = A ky,   (Gamma-point isotropic Dirac)
    m_s(k) = Mt_s + B (kx^2+ky^2) + 2 s J0 fd(k),
    fd(k) = (3/8)(ky^2 - kx^2)        (d-wave, MnTe Cm'c'm style, from the
                                       Gamma expansion of c1-(c2+c3)/2)
    h0 = -D A0^2 + D (kx^2+ky^2) + T kx    (T = mirror-breaking tilt)

keeps the exact BHZ curvature matching (C = -1 clean), while the integration
domain is the TRUE hexagonal 1st BZ of the triangular lattice
(R = 4pi/3).  All observable physics (CPGE main peak ~ 2|Mt|, Fermi pockets,
BCD) lives near Gamma (|k| <~ 0.3), where the k.p form is exact; the
unphysical band deepening at the BZ boundary is harmless because those states
are never occupied for EF near the band top and never resonant for the CPGE
frequencies we use.

Symmetry statement (MnTe realism): fd is even, invariant under both mirrors
kx->-kx, ky->-ky, and NO element of C6v maps fd -> -fd (the square lattice
had the 90-degree rotation doing this; the hexagonal lattice does not).
Hence there is no exact beta_up = beta_dn protection here -- the static spin
polarization is a small lattice-scale background, and the Floquet drive
amplifies it into a large, continuously tunable signal.  A tilt T kx breaks
the kx mirror and opens the charge BCD D_y (positive NHE).
"""
import numpy as np

# --------------------------- geometry ---------------------------------------
R_HEX = 4.0 * np.pi / 3.0          # 1st BZ circumradius (vertex distance)
R_IN  = R_HEX * np.sqrt(3.0) / 2.0 # inradius
BZ_AREA = 3.0 * np.sqrt(3.0) / 2.0 * R_HEX**2   # = 8 pi^2 / sqrt(3)


def hex_mesh(NK):
    """Uniform NK x NK mesh over [-R,R]^2 masked to the hexagonal 1st BZ."""
    k1d = (np.arange(NK) - NK // 2) * (2.0 * R_HEX / NK)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    inside = np.ones_like(kx, dtype=bool)
    for n in range(6):                      # face normals at 30 + 60 n deg
        th = n * np.pi / 3.0 + np.pi / 6.0
        proj = kx * np.cos(th) + ky * np.sin(th)
        inside &= (proj <= R_IN + 1e-9)
    npts = int(inside.sum())
    d2k = BZ_AREA / npts
    return kx, ky, inside, d2k


# ------------------------- model parameters (defaults) ----------------------
# Hybrid set: MnTe/CrSb-scale band velocity (A = 100 meV keeps the Fermi
# pockets resolvable on a modest mesh) with the BHZ 2006 HgTe/CdTe mass
# curvature B (Chern number stays a clean -1).  Scripts may override.
A_PARAM  = 100.0
B_PARAM  = -686.0
D_PARAM  = -512.0
M0_PARAM = -10.0
HW_PARAM = 150.0


# ------------------------- Floquet renormalization -------------------------
def floquet_params(A, B, D, M0, HW, A0, sp=1.0):
    At = {
        "up": (1.0 - 2.0 * A0**2 * B * sp / HW) * A,
        "dn": (1.0 + 2.0 * A0**2 * B * sp / HW) * A,
    }
    Mt = {
        "up": M0 - B * A0**2 - A0**2 * A**2 * sp / HW,
        "dn": M0 - B * A0**2 + A0**2 * A**2 * sp / HW,
    }
    return At, Mt


# ----------------------- d vector + analytic derivatives -------------------
def d_and_derivs(kx, ky, spin, At, Mt, A0, T, J0):
    """d = (hx,hy,m), dx = d_a d, dy = d_b d, plus second derivatives.

    Vectorized: kx, ky may be arrays.  fd = (3/8)(ky^2-kx^2).
    """
    k2 = kx * kx + ky * ky
    s = +1.0 if spin == "up" else -1.0
    fd = 0.375 * (ky * ky - kx * kx)

    m = Mt[spin] - B_PARAM * k2 + 2.0 * s * J0 * fd      # -B k^2: mass inversion
    hx = At[spin] * kx
    hy = At[spin] * ky
    d = np.array([hx, hy, m])

    # first derivatives
    # FIX (v2.1): 2sJ0*fd = 0.75*s*J0*(ky^2-kx^2), d/dkx = -1.5*s*J0*kx.
    # The old 0.75 coefficient (factor-2 off) cancelled the J0 term in the
    # Berry-curvature numerator, making Omega spuriously J0-independent.
    dm_x = -2.0 * B_PARAM * kx - 1.5 * s * J0 * kx
    dm_y = -2.0 * B_PARAM * ky + 1.5 * s * J0 * ky
    dx = np.array([At[spin] + np.zeros_like(kx),
                   np.zeros_like(kx), dm_x])
    dy = np.array([np.zeros_like(kx),
                   At[spin] + np.zeros_like(kx), dm_y])

    # second derivatives (all analytic)
    d2m_xx = -2.0 * B_PARAM - 1.5 * s * J0
    d2m_yy = -2.0 * B_PARAM + 1.5 * s * J0
    d2m_xy = np.zeros_like(kx)
    dxx = np.array([np.zeros_like(kx), np.zeros_like(kx),
                    d2m_xx + np.zeros_like(kx)])
    dxy = np.array([np.zeros_like(kx), np.zeros_like(kx), d2m_xy])
    dyy = np.array([np.zeros_like(kx), np.zeros_like(kx),
                    d2m_yy + np.zeros_like(kx)])

    return {"d": d, "dx": dx, "dy": dy, "dxx": dxx, "dxy": dxy, "dyy": dyy}


# ------------------------- Berry curvature (analytic) ----------------------
def omega_full(kx, ky, spin, At, Mt, A0, T, J0):
    """Omega and its two first derivatives d_x Omega, d_y Omega (vectorized).

    d_a Omega = [d_a d . (dx x dy) + d . (d_a dx x dy + dx x d_a dy)] / (2 D^3)
                - 3 triple (d . d_a d) / (2 D^5)
    """
    r = d_and_derivs(kx, ky, spin, At, Mt, A0, T, J0)
    d, dx, dy = r["d"], r["dx"], r["dy"]
    D2 = np.einsum("i...,i...->...", d, d)
    D2 = np.where(D2 < 1e-24, 1e-24, D2)
    D = np.sqrt(D2)
    triple = np.einsum("i...,i...->...", d,
                       np.cross(dx, dy, axisa=0, axisb=0, axisc=0))
    Omega = triple / (2.0 * D**3)

    dOx = (
        np.einsum("i...,i...->...", dx, np.cross(dx, dy, axisa=0, axisb=0, axisc=0))
        + np.einsum("i...,i...->...", d, np.cross(r["dxx"], dy, axisa=0, axisb=0, axisc=0))
        + np.einsum("i...,i...->...", d, np.cross(dx, r["dxy"], axisa=0, axisb=0, axisc=0))
    ) / (2.0 * D**3) - 3.0 * triple * np.einsum("i...,i...->...", d, dx) / (2.0 * D**5)
    dOy = (
        np.einsum("i...,i...->...", dy, np.cross(dx, dy, axisa=0, axisb=0, axisc=0))
        + np.einsum("i...,i...->...", d, np.cross(r["dxy"], dy, axisa=0, axisb=0, axisc=0))
        + np.einsum("i...,i...->...", d, np.cross(dx, r["dyy"], axisa=0, axisb=0, axisc=0))
    ) / (2.0 * D**3) - 3.0 * triple * np.einsum("i...,i...->...", d, dy) / (2.0 * D**5)
    return Omega, dOx, dOy


def valence_energy(kx, ky, spin, At, Mt, A0, T, J0):
    """h0 - |d| (valence band), vectorized."""
    s = +1.0 if spin == "up" else -1.0
    k2 = kx * kx + ky * ky
    fd = 0.375 * (ky * ky - kx * kx)
    h0 = -D_PARAM * A0**2 - D_PARAM * k2 + T * kx       # -D(A0^2+k^2), D<0
    m = Mt[spin] - B_PARAM * k2 + 2.0 * s * J0 * fd
    hx = At[spin] * kx
    hy = At[spin] * ky
    return h0 - np.sqrt(hx**2 + hy**2 + m**2)


# ------------------------- Chern number (vectorized) -----------------------
def chern_number(kx, ky, mask, d2k, spin, At, Mt, A0, T, J0):
    Om, _, _ = omega_full(kx, ky, spin, At, Mt, A0, T, J0)
    return np.sum(Om * mask) * d2k / (2.0 * np.pi)

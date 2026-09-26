# -*- coding: utf-8 -*-
"""
Paper H -- Script 12: measurable-prediction estimates (order-of-magnitude).

Converts the already-computed model results (official jsons) into SI-scale
experimental estimates.  This is POST-PROCESSING/UNIT CONVERSION only -- it
reads the user's official json files and does not run any new k-mesh physics.

Two channels:

(1) Nonlinear Hall (BCD):  j_a = chi_{abc} E_b E_c, with the Berry-curvature
    dipole D_a [Sodemann & Fu, PRL 115, 216806 (2015)]:
        chi = e^3 tau / (2 hbar^2) * D_a          (2D)
    Our D_model = sum f d_a Omega d2k (units nm; d2k does NOT include
    1/(2 pi)^2, so D_phys = D_model/(2 pi)^2 converted to metres).
    Transverse current:  I_NLH = chi E^2 W ;  V_NLH = I_NLH R,
    R = rho L/(W t) with sample geometry.

(2) CPGE (circular photocurrent) [Moore & Orenstein PRL 105, 026805 (2010);
    de Juan et al. PRB 96, 121115 (2017)]:
        j_c = beta(omega) i[E x E*]_c,   beta ~ e^3/(2 hbar^2 omega^2)
              * sum int d2k/(2pi)^2 Im[r^c r^a] f omega_cv delta(omega-omega_cv)
    Our beta_norm(omega) = sum f Im[r^x r^y] Lor(omega-dE) d2k, with the
    Lorentzian normalized so Lor -> delta in the eta->0 limit.  Taking
    omega_cv ~ omega at resonance,
        beta_phys(omega) = e^3/(2 hbar^2 omega) * beta_norm/(2 pi)^2.
    Field from intensity:  I_opt = eps0 c E0^2 / 2.

All numbers are ORDER-OF-MAGNITUDE estimates with the model's A=100 meV
parameters; systematic O(1) factors (velocity-matrix prefactors, spin sums)
are noted rather than hidden.

Inputs (official jsons, read-only):
  data/hex_bcd_nhe_official_NK400.json  (fallback: hex_bcd_nhe.json)
  data/hex_cpge_official_NK200.json     (fallback: hex_cpge.json)
Outputs:
  terminal table  +  data/observables_estimate.json
"""
import os, json
import numpy as np

# ------------------------------- constants (SI) ----------------------------
E_C   = 1.602176634e-19     # C
HBAR  = 1.054571817e-34     # J s
EPS0  = 8.8541878128e-12    # F/m
CL    = 2.99792458e8        # m/s
MEV   = 1.602176634e-22     # J
NM    = 1e-9                # m

E3_HBAR2 = E_C**3 / HBAR**2          # ~3.7e11  (C^3 s^6 / kg^2 m^4)

# ----------------------- experimental scenario knobs -----------------------
TAU   = float(os.environ.get("PH_TAU_PS", "0.5")) * 1e-12   # s, scattering time
E_DC  = float(os.environ.get("PH_E_DC", "1e5"))             # V/m, bias field for NHE
W_S   = float(os.environ.get("PH_W_UM", "100")) * 1e-6       # sample width
L_S   = float(os.environ.get("PH_L_UM", "100")) * 1e-6       # sample length
T_S   = float(os.environ.get("PH_T_NM", "10")) * NM          # thickness
RHO_METAL  = 1e-7                    # Ohm m (CrSb-like metal)
RHO_SEMI   = 1e-3                    # Ohm m (MnTe doped semiconductor, order)
I_OPT = float(os.environ.get("PH_IOPT_Wcm2", "100")) * 1e4   # W/m^2 (100 W/cm^2)

def D_phys(D_model_nm):
    return abs(D_model_nm) / (2.0 * np.pi)**2 * NM           # -> metres

def chi_from_D(D_model_nm):
    return E3_HBAR2 * TAU / 2.0 * D_phys(D_model_nm)

def beta_phys(beta_norm, omega_meV):
    om = omega_meV * MEV / HBAR                      # rad/s
    return E3_HBAR2 / (2.0 * om) * beta_norm / (2.0 * np.pi)**2

def main():
    out = {"constants": {"tau_s": TAU, "E_DC_Vpm": E_DC, "W_m": W_S,
                         "L_m": L_S, "t_m": T_S, "I_opt_Wpm2": I_OPT},
           "nhe": {}, "cpge": {}}

    # ================= (1) nonlinear Hall ==================================
    bcd_path = ("data/hex_bcd_nhe_official_NK400.json"
                if os.path.exists("data/hex_bcd_nhe_official_NK400.json")
                else "data/hex_bcd_nhe.json")
    bcd = json.load(open(bcd_path))
    rows = bcd["ef_scan"]
    # pick the T=0, tilt=30 single-spin-window row (delta=3.0) and the
    # T=30 (room) row at the same depth
    t0  = [r for r in rows if r["T"] == 0.0  and r["TILT"] == 30.0 and r["delta"] == 3.0]
    t30 = [r for r in rows if r["T"] == 30.0 and r["TILT"] == 30.0 and r["delta"] == 3.0]
    print("== Nonlinear Hall (BCD) estimates ==")
    for tag, sel in (("T=0 single-spin window (Dx_up only)", t0),
                     ("T=30 meV room temp (Dx_tot)", t30)):
        if not sel:
            continue
        r = sel[0]
        for lbl, Dk in (("Dx_up", "Dx_up"), ("Dx_dn", "Dx_dn"),
                        ("Dx_tot", "Dx_tot"), ("Dx_spin", "Dx_spin")):
            chi = chi_from_D(r[Dk])
            I_nl = chi * E_DC**2 * W_S
            # resistance for the two material classes
            Rm = RHO_METAL * L_S / (W_S * T_S)
            Rs = RHO_SEMI  * L_S / (W_S * T_S)
            rec = {"D_model_nm": r[Dk], "D_phys_m": D_phys(r[Dk]),
                   "chi_2D_Am_V2": chi, "I_NLH_A": I_nl,
                   "V_NLH_metal_uV": I_nl * Rm * 1e6,
                   "V_NLH_semi_uV": I_nl * Rs * 1e6}
            out["nhe"][f"{tag}|{lbl}"] = rec
            print(f"  [{tag}] {lbl}: D_model={r[Dk]:+.2f} nm -> "
                  f"D_phys={rec['D_phys_m']:.2e} m, "
                  f"chi={chi:.2e} A m/V^2, I_NLH={I_nl:.2e} A "
                  f"@E={E_DC:.0e} V/m, W={W_S*1e6:.0f} um")
            print(f"      V_NLH ~ {rec['V_NLH_metal_uV']:.1f} uV (CrSb-like metal), "
                  f"~ {rec['V_NLH_semi_uV']:.1f} uV (MnTe-like doped semi)")

    # ================= (2) CPGE ============================================
    cp_path = ("data/hex_cpge_official_NK200.json"
               if os.path.exists("data/hex_cpge_official_NK200.json")
               else "data/hex_cpge.json")
    cp = json.load(open(cp_path))
    print("\n== CPGE photocurrent estimates (M=-10, A0=0.06) ==")
    res = cp["results"]["M-10.0"]["A0_0.060"]
    om = np.array(res["omega"]); bu = np.array(res["beta_up"])
    bd = np.array(res["beta_dn"]); bt = np.array(res["beta_tot"])
    E0sq = 2.0 * I_OPT / (EPS0 * CL)                     # V^2/m^2
    for lbl, b in (("beta_up", bu), ("beta_dn", bd), ("beta_tot", bt)):
        i = int(np.argmax(np.abs(b)))
        bp = beta_phys(float(b[i]), float(om[i]))
        j  = bp * E0sq                                   # A/m^2
        I_cp = j * W_S * T_S                             # A (through width*thick)
        rec = {"omega_meV": float(om[i]), "beta_norm": float(b[i]),
               "beta_phys_A_V2": bp, "E0sq_V2_m2": E0sq,
               "j_cpge_A_m2": j, "I_cpge_A": I_cp}
        out["cpge"][lbl] = rec
        print(f"  {lbl}: peak @ {om[i]:.1f} meV, beta_norm={b[i]:+.3e} -> "
              f"beta_phys={bp:.2e} A/V^2, j_cpge={j:.2e} A/m^2, "
              f"I_cpge={I_cp:.2e} A @ {I_OPT/1e4:.0f} W/cm^2")
    # spin-polarized CPGE at the strong-polarization point M=-16, A0=0.12
    try:
        res2 = cp["results"]["M-16.0"]["A0_0.120"]
        om2 = np.array(res2["omega"]); bu2 = np.array(res2["beta_up"])
        bd2 = np.array(res2["beta_dn"])
        iu = int(np.argmax(np.abs(bu2))); idn = int(np.argmax(np.abs(bd2)))
        ju_ = beta_phys(float(bu2[iu]), float(om2[iu])) * E0sq
        jd_ = beta_phys(float(bd2[idn]), float(om2[idn])) * E0sq
        out["cpge"]["M-16_A0.12_up"] = {"omega_meV": float(om2[iu]),
                                          "j_A_m2": ju_}
        out["cpge"]["M-16_A0.12_dn"] = {"omega_meV": float(om2[idn]),
                                          "j_A_m2": jd_}
        print(f"  [M=-16, A0=0.12 near-full polarization] j_up={ju_:.2e} A/m^2 "
              f"@ {om2[iu]:.1f} meV vs j_dn={jd_:.2e} A/m^2 @ {om2[idn]:.1f} meV")
    except KeyError:
        print("  (M=-16 / A0=0.12 row not in json -- skipped)")

    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/observables_estimate.json", "w"), indent=2)
    print("\nSaved: data/observables_estimate.json")
    print("Note: order-of-magnitude only; O(1) prefactors (spin sums, ")
    print("velocity-matrix normalization) are omitted; tau and I_opt are knobs.")

if __name__ == "__main__":
    main()

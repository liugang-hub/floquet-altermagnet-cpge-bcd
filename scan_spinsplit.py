"""Spin-splitting scaling of the altermagnet band tops vs J0 (analysis tool).

Question: the single-spin Fermi-window width is Delta = top_up - top_dn at
A0 = 0.06. How does Delta scale with the altermagnet coupling J0?

Band tops are LOCAL quantities (max of valence band over the mesh), so a
moderate NK is sufficient -- this is analysis, not production numerics.
"""
import os
import numpy as np
import hex_model as hm

A, B, D, M0 = hm.A_PARAM, hm.B_PARAM, hm.D_PARAM, hm.M0_PARAM
HW = hm.HW_PARAM
A0 = 0.06
NK = int(os.environ.get("PH_NK", "96"))
J0S = [float(x) for x in
       os.environ.get("PH_J0S", "0,40,80,120,160,200,240").split(",")]

kx, ky, mask, d2k = hm.hex_mesh(NK)


def bandtops(J0, TILT):
    At, Mt = hm.floquet_params(A, B, D, M0, HW, A0)
    out = {}
    for spin in ("up", "dn"):
        Ev = hm.valence_energy(kx, ky, spin, At, Mt, A0, TILT, J0)
        Ew = np.where(mask, Ev, -1e9)
        i = int(np.argmax(Ew))
        out[spin] = float(Ew.ravel()[i])
        out[spin + "k"] = (float(kx.ravel()[i]), float(ky.ravel()[i]))
    return out


print(f"hexagonal BZ, A0={A0}, NK={NK}")
print("J0   | tilt=30: top_up  top_dn  Delta | k*_up      k*_dn    | tilt=0 Delta")
js = []
ds30, ds0 = [], []
for J0 in J0S:
    t30 = bandtops(J0, 30.0)
    t0 = bandtops(J0, 0.0)
    d30 = t30["up"] - t30["dn"]
    d0 = t0["up"] - t0["dn"]
    js.append(J0); ds30.append(d30); ds0.append(d0)
    print(f"{J0:4.0f} | {t30['up']:+8.3f} {t30['dn']:+8.3f} {d30:+8.3f} | "
          f"({t30['upk'][0]:+.3f},{t30['upk'][1]:+.3f}) ({t30['dnk'][0]:+.3f},{t30['dnk'][1]:+.3f}) | {d0:+8.3f}")

# linear fits
for name, ds in (("tilt=30", ds30), ("tilt=0", ds0)):
    p = np.polyfit(js, ds, 1)
    res = np.array(ds) - np.polyval(p, js)
    print(f"Delta(J0) [{name}]: slope = {p[0]:+.4f} meV/meV, intercept = {p[1]:+.3f} meV, max resid = {np.max(np.abs(res)):.3f} meV")

# analytic attribution: Delta = 2*|dMt| + 4*J0*|fd(k*)|
At, Mt = hm.floquet_params(A, B, D, M0, HW, A0)
dMt = Mt["up"] - Mt["dn"] if isinstance(Mt, dict) else None
print("Mt_up, Mt_dn =", Mt)
print("dMt = Mt_up - Mt_dn =", dMt, " -> 2|dMt| =", 2 * abs(dMt))

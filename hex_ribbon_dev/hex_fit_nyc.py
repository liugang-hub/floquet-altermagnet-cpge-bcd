# -*- coding: utf-8 -*-
r"""
hex_fit_nyc.py -- B3a 收官: 标度律 Ny_c0(A0) = C * xi_up(A0) 的最小二乘拟合
============================================================================
读 data/hex_rib_width_scan.json (B3a 实测), 提取每 A0 的 (xi_up_rows, Ny_c0),
做两种拟合并报告:
  (1) 逐点比值 r = Ny_c0 / xi_up          (粗检: 是否 ~2.4 常数)
  (2) 无截距 LSQ:  C0 = sum(x*y)/sum(x^2)  (标度律严格形式)
  (3) 带截距 LSQ:  y = C*x + b (np.polyfit deg1), 报 C, b, R^2
      (若 b 接近 0 且 R^2 接近 1 => 纯标度成立, C 即"需容纳的穿透深度数")

纯 numpy 后处理, 不 import kwant.  运行 (kwant2 终端, hex_ribbon 目录):
      python hex_fit_nyc.py
输出: 终端表格 + data/hex_fit_nyc.json
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "hex_rib_width_scan.json")
OUT = os.path.join(HERE, "data", "hex_fit_nyc.json")

with open(SRC, encoding="utf-8") as f:
    raw = json.load(f)

rows = []
for k, v in raw["part1"].items():
    xi = v.get("xi_up_rows")
    nc = v.get("Ny_c0")
    if xi is None or nc is None:
        continue
    rows.append((float(k), float(xi), float(nc)))

print("== Ny_c0(A0) ~ C*xi_up(A0)  拟合 (%d 点) ==" % len(rows))
print("  A0      xi_up   Ny_c0   r=Ny_c0/xi")
for a0, xi, nc in rows:
    print("  %6.3f  %5.0f  %4d   %6.3f" % (a0, xi, nc, nc / xi))

x = np.array([r[1] for r in rows], dtype=float)
y = np.array([r[2] for r in rows], dtype=float)

C0 = float(np.sum(x * y) / np.sum(x * x))
c1, b1 = np.polyfit(x, y, 1)
yhat = c1 * x + b1
ss_res = float(np.sum((y - yhat) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

print("\n[无截距]  Ny_c0 = %.4f * xi_up" % C0)
print("[带截距]  Ny_c0 = %.4f * xi_up + %.4f    R^2 = %.6f" % (c1, b1, r2))
print("提示: |b| < 0.5 步长(8行) 视为纯标度; C ~ 2.4 => 器件窗需 ~2.4 个穿透深度")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"points": [{"A0": a, "xi_up": xi, "Ny_c0": nc,
                           "ratio": round(nc / xi, 4)} for a, xi, nc in rows],
               "fit": {"C0_noconst": round(C0, 4),
                       "C": round(float(c1), 4),
                       "b": round(float(b1), 4),
                       "R2": round(r2, 6)}},
              f, ensure_ascii=False, indent=1)
print("\n[ok] %s" % OUT)

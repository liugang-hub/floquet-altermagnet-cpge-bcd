# Floquet-Engineered Helical Spin Polarization and Nonlinear Hall Effect in Altermagnetic Hexagonal Ribbons

Code and data accompanying the manuscript submitted to *Physical Review B* (2026).

## Repository layout

- `*.py` (root) — bulk-response production scripts: tight-binding model library (`hex_model.py`), CPGE, BCD/NHE, tilt/frequency/amplitude response surfaces, dense chemical-potential scans, temperature and Lorentzian-broadening scans.
- `data/` — archived numerical output (JSON) of the bulk calculations.
- `hex_ribbon_dev/` — kwant-based ribbon device scripts (two-terminal transport, dephasing, Anderson disorder, edge roughness, bond-current visualization, width scaling), with its own `hex_ribbon_dev/data/` folder.
- `figures/` — the seven manuscript figures, for browsing.

## Figure-to-script-to-data map

| Manuscript figure | Content | Plotting script | Data |
|---|---|---|---|
| Fig. 1 (`fig2_cpge.png`) | Spin-resolved CPGE | `hex_cpge.py` | `data/hex_cpge_official_NK200.json` |
| Fig. 2 (`fig3_bcd_nhe.png`) | BCD/NHE single-spin window | `redraw_bcd_panel.py` | `data/hex_bcd_nhe.json` |
| Fig. 3 (`fig4_ef_dense.png`) | Dense chemical-potential scan | `redraw_efdense_panel.py` | `data/hex_ef_dense.json` |
| Fig. 4 (`fig5_response.png`) | Tilt/amplitude/frequency response surfaces | `hex_bcd_paperfig.py` | `data/hex_bcd_surface_*.json` |
| Fig. 5 (`fig6_device.png`) | Ribbon device transport | `hex_ribbon_dev/hex_rib_dp_paperfig.py` | `hex_ribbon_dev/data/hex_rib_disorder_dp_disdp.json` |
| Fig. 6 (`hex_bcd_temp_temp.png`) | Full temperature curve with broadening (32 configurations) | `hex_bcd_temp_fig.py` | `data/hex_bcd_temp_temp.json` |
| Fig. 7 (`hex_rib_edge_rough_er.png`) | Edge-roughness robustness (960 devices) | `hex_ribbon_dev/hex_rib_edge_rough_fig.py` | `hex_ribbon_dev/data/hex_rib_edge_rough_er.json` |

The `redraw_*.py` and `*_fig.py` scripts regenerate the manuscript figures directly from the archived JSON (no recompute). The remaining scripts are the production codes that generated those JSON files.

## Requirements

- Python 3, NumPy, Matplotlib (all bulk scripts)
- `hex_ribbon_dev/` additionally requires [kwant](https://kwant-project.org/)

## License

MIT (code). Manuscript figures and text are courtesy of the authors; please cite the paper if you use them.

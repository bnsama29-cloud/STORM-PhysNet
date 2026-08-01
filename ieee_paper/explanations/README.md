# STORM-PhysNet — IEEE Manuscript Package

## Contents

| Path | Description |
|------|-------------|
| `conference/storm_physnet_conference.tex` | **IEEE conference** draft (target ~6 pages) |
| `access/storm_physnet_access.tex` | **IEEE Access** draft (full results) |
| `figures/` | Place all plot PNGs/PDFs here (same folder for both) |

## Figure files required (copy from Drive `nb3_outputs/plots` + `nb4_outputs/plots`)

```
fig1_horizon_pe.png
fig2_all_vs_storm.png
fig3_storm_weight_sweep.png
fig4_ablation_multi.png          # TBD — after multi-seed ablation + eval
fig5_timeseries_storm_6h.png
fig6_delay_hist.png
fig7_gate_activation.png
fig8_grasp_domain_gap.png
fig9_grasp_horizon.png
fig10_grasp_fewshot.png
fig_timeseries_storm.png         # optional 3-panel
```

Copy them into `figures/` (or keep paths relative as in the `.tex` files: `../figures/...`).

## Build

```bash
# Conference
cd conference
pdflatex storm_physnet_conference.tex
bibtex storm_physnet_conference   # if using .bib; this draft uses thebibliography
pdflatex storm_physnet_conference.tex
pdflatex storm_physnet_conference.tex

# Access
cd ../access
pdflatex storm_physnet_access.tex
pdflatex storm_physnet_access.tex
```

Or use Overleaf: upload the folder, set main file, upload figures.

## TBD before submission

1. Finish multi-seed `no_delay` / `no_physics` training → run NB3+ABL eval.
2. Replace **TBD** rows in Table I and insert `fig4_ablation_multi.png`.
3. Author names, affiliations, funding, acknowledgments.
4. Double-check bibliography DOIs against the actual papers you cite.
5. Conference: drop optional figures if over page limit (see comments in `.tex`).

## What is already filled

- GOES multi-seed PE numbers (storm_bz best, ensemble 0.718)
- Single-seed baselines
- GRASP transfer table
- Interpretability (τ ≈ 1.09 h)
- All figure includes except ablation multi (placeholder)

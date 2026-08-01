# Git + figure checklist

## Main repo (STORM-PhysNet)

1. README PE table = paper Table I only (0.669/0.674 vs TF 0.648/0.612).
2. Feature count: 16 (or "solar-wind features"), not 14.
3. tau ≈ 1.5 h on final checkpoints.
4. Params ≈ 3.4e5 (STORM) vs ≈ 8.4e5 (Transformer).
5. Code availability link matches this repo.
6. Add CITATION.cff or bib snippet for the paper title.
7. Archive/private old repos or banner: "Superseded — ignore old PE 0.42/0.9 claims."

## Figures (Overleaf `figures/`)

Required Access names (after rename):
- fig_system_architecture.png
- fig1_horizon_pe.png
- fig4_ablation_multi.png
- fig_timeseries_storm.png
- fig6_delay_hist.png
- fig7_gate_activation.png
- fig_feature_importance.png
- fig_residual_storm_bz.png
- fig8_grasp_domain_gap.png
- fig_dashboard_storm_flux.png

Architecture diagram edits:
- 14 → 16 (or remove fixed width)
- Remove hard "aleatoric s^2" as main output
- Soften gate label to Bz-conditioned
- Align loss box with paper loss text

Dashboard:
- GSAT-19 naming consistent with paper
- Caption: illustration only

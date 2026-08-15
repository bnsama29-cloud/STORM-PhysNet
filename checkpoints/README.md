# Checkpoints

Best weights for seeds 42–56 used in the papers.

Layout: checkpoints/<model>/seed_<id>/*_best.pt

Models and seed coverage:
- storm_bz: 42–56 (15 seeds)
- transformer: 42–56 (15 seeds, d_model=64, 3 layers, 4 heads — not capacity-matched)
- transformer_matched: 42–56 (15 seeds, d_model=128, 2 layers, 4 heads — capacity-matched control)
- lstm: 42–56 (15 seeds)
- storm_no_delay: 42–56 (15 seeds)
- storm_no_gate: 42–56 (15 seeds)
- storm_no_physics: 42–56 (15 seeds)
- bagged_tf: 42–56 (15 seeds)
- grasp: 42–56 (15 seeds, fine-tuned heads)
- alt_gates/storm_cathode: 42–56 (15 seeds, JSON metrics only)
- alt_gates/storm_cathode_spec: 42–56 (15 seeds, JSON metrics only)
- alt_gates/storm_radiotrophic: 42–56 (15 seeds, JSON metrics only)

Transformer baseline: d_model=64, 3 layers, 4 heads (not capacity-matched to STORM).
Transformer matched: d_model=128, 2 layers, 4 heads (~1.19e6 params; capacity-matched control).
Paper PE tables are also summarized in results/*.csv and results/*.json.

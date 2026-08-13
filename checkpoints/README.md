# Checkpoints

Best weights for seeds 42–56 used in the papers.

Layout: checkpoints/<model>/seed_<id>/*_best.pt

Models and seed coverage:
- storm_bz: 42–56 (15 seeds)
- transformer: 42–46 (5 seeds; 47–56 not yet packaged)
- lstm: 42–56 (15 seeds)
- storm_no_delay: 42–56 (15 seeds)
- storm_no_gate: 42–56 (15 seeds)
- storm_no_physics: 42–56 (15 seeds)
- bagged_tf: 42–56 (15 seeds)
- grasp: 42–56 (15 seeds, fine-tuned heads)

Transformer baseline: d_model=64, 3 layers, 4 heads (not capacity-matched to STORM).
Paper PE tables are also summarized in results/*.csv.

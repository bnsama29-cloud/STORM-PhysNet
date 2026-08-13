# Checkpoints

Best weights for seeds 42–56 used in the papers.

Layout: checkpoints/<model>/seed_<id>/*_best.pt

Models: storm_bz, transformer, lstm, storm_no_delay, storm_no_gate,
        storm_no_physics, grasp (fine-tuned heads).

Transformer baseline: d_model=64, 3 layers, 4 heads (not capacity-matched to STORM).
Paper PE tables are also summarized in results/*.csv.

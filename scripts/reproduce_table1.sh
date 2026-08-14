#!/bin/bash
# Reproduction script for Table 1 (Multi-seed test results)

echo "Reproducing STORM-PhysNet results (Table I)"

# 1. Main Training (STORM-Bz, Transformer baseline)
python scripts/run_training.py --config configs/config.yaml

# 2. Ablation Studies (No-delay, No-physics, No-gate)
python scripts/02_train_ablations_baselines.py

# 3. Capacity-Matched Transformer (Appendix/Access Table)
python scripts/account9_final_matched_tf.py

# 4. Generate Final Metrics & Bagged Ensemble Output
python scripts/final_ablation_eval.py
python scripts/04_ensemble_hybrid_eval.py
python scripts/compute_matched_tf_bagged.py

echo "Check results/ablation_final_table.csv and results/summary.json for the final numbers."

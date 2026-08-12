# Results

- `summary.json` — curated numbers matching the papers (main table, wider-delay, bagged Transformer).
- `all_results.csv`, `wider_delay_results.csv`, `bagged_tf_results.csv` — seed-level outputs from the evaluation runs.
- `ablation_final_table.csv` — ablation summary.
- `wider_delay_pe6h.png` — figure used in the Access paper.

**Note:** Full model checkpoints (`.pt`) are not stored in this repo (size).  
Training code + data + these CSVs are sufficient to verify reported PE values.
Synthetic utilities under `src/data/synthetic_generator.py` and
`src/data/storm_augmentor.py` are **not** used in the paper training pipeline.

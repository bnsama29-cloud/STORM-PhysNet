# Results

| File | Contents |
|------|----------|
| `summary.json` | Curated numbers matching the conference + Access papers |
| `all_results.csv` | Seed-level evaluation outputs |
| `wider_delay_results.csv` | Wider delay upper-bound ablation |
| `bagged_tf_results.csv` | Bagged Transformer control (16 seeds) |
| `ablation_final_table.csv` | Ablation summary |
| `wider_delay_pe6h.png` | Access paper figure |

Checkpoints (`.pt`) are not released in this repository.  
Code + `datasets/` + these CSVs are enough to verify the reported PE values.

**Note:** `src/data/synthetic_generator.py` and `src/data/storm_augmentor.py` are development utilities only and are **not** used in the published training pipeline.
```
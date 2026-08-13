# Results

| File | Contents |
|------|----------|
| `summary.json` | Curated numbers matching the conference + Access papers |
| `all_results.csv` | Seed-level evaluation outputs |
| `wider_delay_results.csv` | Wider delay upper-bound ablation |
| `bagged_tf_results.csv` | Bagged Transformer control (16 seeds) |
| `ablation_final_table.csv` | Ablation summary |
| `wider_delay_pe6h.png` | Access paper figure |
| `grasp_metrics_summary.csv` | GRASP zero-shot / fine-tune results |

Checkpoints (`.pt`) are not released in this repository.  
Code + `datasets/` + these CSVs are enough to verify the reported PE values.

**Note:** The wider-delay 2.0 h bound used 16 seeds; all other bounds and main tables use 15 seeds.

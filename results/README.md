# Results

| File | Contents |
|------|----------|
| `summary.json` | Curated numbers matching the conference + Access papers |
| `all_results.csv` | Seed-level evaluation outputs |
| `wider_delay_results.csv` | Wider delay upper-bound ablation |
| `bagged_tf_results.csv` | Bagged Transformer control (16 seeds) |
| `transformer_matched_summary.json` | Capacity-matched Transformer mean PE (15 seeds) |
| `transformer_matched_bagged_pe.json` | Capacity-matched Transformer bagged PE (15 seeds) |
| `transformer_matched_seed_pe.csv` | Capacity-matched Transformer per-seed PE |
| `matched_tf_seeds/seed_*.json` | Per-seed evaluation JSONs for matched Transformer |
| `ablation_final_table.csv` | Ablation summary |
| `wider_delay_pe6h.png` | Access paper figure |
| `grasp_metrics_summary.csv` | GRASP zero-shot / fine-tune results |

Checkpoints (`.pt`) are not released in this repository.  
Code + `datasets/` + these CSVs are enough to verify the reported PE values.

**Note:** The wider-delay 2.0 h bound used 16 seeds; all other bounds and main tables use 15 seeds.

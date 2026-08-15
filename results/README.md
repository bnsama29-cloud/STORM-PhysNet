# Results

| File | Contents |
|------|----------|
| `summary.json` | Curated numbers matching the conference + Access papers |
| `all_results.csv` | Seed-level evaluation outputs |
| `wider_delay_results.csv` | Wider delay upper-bound ablation |
| `bagged_tf_results.csv` | Bagged Transformer control (16 seeds) |
| `transformer_matched_summary.json` | Architecture-matched Transformer mean PE (15 seeds) |
| `transformer_matched_bagged_pe.json` | Architecture-matched Transformer bagged PE (15 seeds) |
| `transformer_matched_seed_pe.csv` | Architecture-matched Transformer per-seed PE |
| `ablation_final_table.csv` | Ablation summary |
| `wider_delay_pe6h.png` | Access paper figure |
| `grasp_metrics_summary.csv` | GRASP zero-shot / fine-tune results |
| `matched_tf_grasp.csv` | GRASP fine-tuning training logs (Architecture-matched Transformer) |
| `matched_tf_noise.csv` | Noise robustness results (Architecture-matched Transformer) |
| `matched_tf_pers.csv` | Operational Persistence results |

15 architecture-matched Transformer checkpoints (`.pt`) **are** released in this repository in the `checkpoints/transformer_matched/` directory.  
Code + `datasets/` + these CSVs are enough to verify the reported PE values.

**Note:** The wider-delay 2.0 h bound used 16 seeds; all other bounds and main tables use 15 seeds.

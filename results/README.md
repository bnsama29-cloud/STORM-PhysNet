# STORM-PhysNet Results

This directory contains the final evaluation exports and metric summaries for the STORM-PhysNet project.

### Core PE Tables & Seed Results
- `table_main_means.csv` - Mean Performance across 15 seeds for baseline vs STORM.
- `table_bagged.csv` - Ensemble Bagged prediction efficiency (all architectures).
- `all_seed_results_full.csv` - Complete 15-seed data block.
- `BAGGED_*.json` - The raw JSON exports for the bagged results by model type.
- `ensemble_*.csv`/`.json` - Metrics for the validation-selected linear ensemble ($\alpha^*=0.3$).
- `hybrid_per_seed.csv` - Per-seed results for the experimental hybrid model.

### GRASP Domain Transfer
- `grasp_all_seeds.csv` - All 15 seeds evaluated on GSAT-19 GRASP data.
- `grasp_summary.csv` - Summary of the zero-shot vs fine-tuning domain transfer performance.
- `table_grasp_storm_bz.csv` - Formatted GRASP table export.

### Other Folders
- `paper_export/` - Additional exported paper tables (duplicates of root CSVs/JSONs kept for legacy).
- `seeds/` - Raw predictions and loss traces per individual seed (if available locally).

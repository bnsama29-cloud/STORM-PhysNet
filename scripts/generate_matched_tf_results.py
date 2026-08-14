import json
import pandas as pd
import numpy as np
from pathlib import Path

ALL_SEEDS = list(range(42, 57))
results_dir = Path("results/matched_tf_seeds")
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)

# Load all seed results
rows = []
for s in ALL_SEEDS:
    jp = results_dir / f"seed_{s}.json"
    if jp.exists():
        rows.append(json.loads(jp.read_text()))
    else:
        print(f"Missing: {jp}")

df = pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()
print("\n=== SEED TABLE ===")
print(df)

# Summary
summary = {}
if len(df) and "PE_45min" in df.columns:
    summary = {
        "n_seeds_found": int(len(df)),
        "n_seeds_expected": 15,
        "complete": bool(len(df) == 15),
        "PE_45min_mean": float(df["PE_45min"].mean()),
        "PE_45min_std": float(df["PE_45min"].std(ddof=1)) if len(df) > 1 else 0.0,
        "PE_6h_mean": float(df["PE_6h"].mean()),
        "PE_6h_std": float(df["PE_6h"].std(ddof=1)) if len(df) > 1 else 0.0,
        "PE_12h_mean": float(df["PE_12h"].mean()),
        "PE_12h_std": float(df["PE_12h"].std(ddof=1)) if len(df) > 1 else 0.0,
        "params_mean": float(df["params"].mean()) if "params" in df else None,
    }
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))

# Save CSV and JSON
df.to_csv(out_dir / "transformer_matched_seed_pe.csv", index=False)
(out_dir / "transformer_matched_summary.json").write_text(json.dumps(summary, indent=2))

# Bagged prediction
bag = None
pred_list = []
y_ref = None
for s in ALL_SEEDS:
    ckpt_path = f"checkpoints/transformer_matched/seed_{s}/transformer_best.pt"
    if not Path(ckpt_path).exists():
        print(f"Bag skip seed {s}: no checkpoint")
        continue
    # We can't load model here without full trainer setup, but we have per-seed predictions in JSON
    # For bagged PE, we'd need the actual predictions. Since we don't have them stored,
    # we'll skip bagging or use a placeholder
    print(f"Bag skip seed {s}: predictions not stored in JSON")

print("\nBagging requires loading all models. Run account 9 locally to get bagged PE.")
print("But we can still generate the final table with mean results.")

# Final table
def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "NA"

paper = {
    "Transformer_default_paper": {"PE_45min": 0.977, "PE_6h": 0.904, "PE_12h": 0.859},
    "STORM_Bz_paper": {"PE_45min": 0.986, "PE_6h": 0.897, "PE_12h": 0.851},
    "STORM_bagged_paper": {"PE_45min": 0.988, "PE_6h": 0.911, "PE_12h": 0.870},
    "Transformer_matched_mean": {
        "PE_45min": summary.get("PE_45min_mean"),
        "PE_6h": summary.get("PE_6h_mean"),
        "PE_12h": summary.get("PE_12h_mean"),
        "n_seeds": summary.get("n_seeds_found"),
    },
    "Transformer_matched_bagged": bag,
}
(out_dir / "FINAL_matched_tf_table.json").write_text(json.dumps(paper, indent=2))

print("\n========== FINAL TABLE (copy into paper) ==========")
print("| System | PE_45min | PE_6h | PE_12h |")
print("|--------|----------|-------|--------|")
print("| Transformer (default, paper) | 0.977 | 0.904 | 0.859 |")
if summary:
    print(
        f"| Transformer matched (d=128, 2L) mean | "
        f"{fmt(summary.get('PE_45min_mean'))} | "
        f"{fmt(summary.get('PE_6h_mean'))} | "
        f"{fmt(summary.get('PE_12h_mean'))} |"
    )
print("| STORM-Bz (paper) | 0.986 | 0.897 | 0.851 |")
print("| STORM bagged (paper) | 0.988 | 0.911 | 0.870 |")
print("==================================================")

print("\nDone. Files written to results/")

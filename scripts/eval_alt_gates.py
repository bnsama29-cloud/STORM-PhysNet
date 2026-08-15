import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOTS = [
    Path("f:/Downloads/ieee_final_fixed/results/alt_gates"),
]

rows = []
for root in ROOTS:
    if not root.exists():
        continue
    for p in root.rglob("seed_*.json"):
        rows.append(json.loads(p.read_text()))

if not rows:
    raise SystemExit("No JSONs — merge account zips first")

df = pd.DataFrame(rows).drop_duplicates(subset=["model", "seed"], keep="last")

# tolerate old key PE_45min if any leftover files
if "PE_1h" not in df.columns and "PE_45min" in df.columns:
    df["PE_1h"] = df["PE_45min"]

print("Rows:", len(df))
print(df.groupby("model")["seed"].count())

summary = df.groupby("model")[["PE_1h", "PE_6h", "PE_12h"]].agg(["mean", "std"])
print(summary)
summary.to_csv("alt_gates_summary.csv")

# plot PE_6h for the 3 variants
ax = summary["PE_6h"]["mean"].plot.bar(yerr=summary["PE_6h"]["std"], capsize=4, title="Alt-Gates PE_6h")
plt.tight_layout()
plt.savefig("fig_alt_gates_pe6h.png")
print("Saved alt_gates_summary.csv and fig_alt_gates_pe6h.png")

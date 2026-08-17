
import os, sys, json, shutil
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. SETUP PATHS
# ---------------------------------------------------------
ROOT_DIR = Path(r"f:\Downloads\ieee_final_fixed")
ZIP_DIR = ROOT_DIR / "kaggle_outputs"

RESULTS_DIR = ROOT_DIR / "results"
CKPT_DIR = ROOT_DIR / "checkpoints"
OUT_DIR = ROOT_DIR / "paper_export"
FIG_DIR = ROOT_DIR / "figures"

# Clean up any old data to prevent mixing
shutil.rmtree(RESULTS_DIR, ignore_errors=True)
shutil.rmtree(CKPT_DIR, ignore_errors=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. EXTRACT ZIPS
# ---------------------------------------------------------
zips_found = list(ZIP_DIR.glob("*.zip"))
if not zips_found:
    print(f"ERROR: No zip files found in {ZIP_DIR}!")
    print("Please download your Kaggle outputs and put them in this folder.")
    sys.exit(1)

for z in zips_found:
    print(f"Extracting {z.name}...")
    if "results" in z.name:
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(RESULTS_DIR)
    elif "ckpt" in z.name:
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(CKPT_DIR)

# ---------------------------------------------------------
# 3. RUN CELL 2 (AGGREGATION)
# ---------------------------------------------------------
print("\n--- Running Data Aggregation ---")

def load_all_json():
    rows = []
    for p in list(RESULTS_DIR.rglob("seed_*.json")):
        try:
            d = json.loads(p.read_text())
            rows.append(d)
        except Exception as e:
            print("skip", p, e)
    return pd.DataFrame(rows)

df = load_all_json()
if df.empty:
    raise SystemExit("No seed_*.json found! Extraction failed or zips are empty.")

df.to_csv(OUT_DIR / "all_seed_results.csv", index=False)
print(f"Loaded {len(df)} seed rows (Should be 15 seeds * 10 models = 150 rows)")

# Means and Stds
PE_COLS = [c for c in ["PE_1h", "PE_6h", "PE_12h", "PE_pers_1h"] if c in df.columns]
ORDER = [
    "lstm", "transformer", "storm_bz",
    "storm_no_delay", "storm_no_physics", "storm_no_gate",
    "transformer_matched",
    "storm_cathode", "storm_cathode_spec", "storm_radiotrophic",
]

summary = df.groupby("name")[PE_COLS].agg(["mean", "std", "count"])
summary.to_csv(OUT_DIR / "summary_mean_std.csv")

flat = df.groupby("name")[PE_COLS].mean().reindex([n for n in ORDER if n in df["name"].unique()])
flat.to_csv(OUT_DIR / "table_main_means.csv")

# Proxy for True Bagging (Seed Mean)
if "transformer_matched" in df["name"].values:
    sub = df[df["name"] == "transformer_matched"]
    bagged_proxy = {
        "n_seeds": int(len(sub)),
        "PE_1h_mean": float(sub["PE_1h"].mean()),
        "PE_6h_mean": float(sub["PE_6h"].mean()),
        "PE_12h_mean": float(sub["PE_12h"].mean()),
    }
    (OUT_DIR / "transformer_matched_seed_mean.json").write_text(json.dumps(bagged_proxy, indent=2))

# Figures
name_map = {
    "storm_bz": "STORM-Bz", "storm_cathode": "RDG", "storm_cathode_spec": "RDG-S",
    "storm_radiotrophic": "SDG", "transformer": "Transformer", "lstm": "LSTM",
    "transformer_matched": "Transformer matched", "storm_no_delay": "No-Delay",
    "storm_no_physics": "No-Physics", "storm_no_gate": "No-Gate",
}

plot_names = [n for n in ["lstm", "transformer", "storm_bz", "transformer_matched", "storm_cathode", "storm_radiotrophic"] if n in df["name"].unique()]
if plot_names:
    means = df.groupby("name")[["PE_1h", "PE_6h", "PE_12h"]].mean().reindex(plot_names)
    stds = df.groupby("name")[["PE_1h", "PE_6h", "PE_12h"]].std().reindex(plot_names)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(plot_names))
    w = 0.25
    for i, h in enumerate(["PE_1h", "PE_6h", "PE_12h"]):
        ax.bar(x + (i - 1) * w, means[h].values, w, yerr=stds[h].values, capsize=3, label=h.replace("PE_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([name_map.get(n, n) for n in plot_names], rotation=20, ha="right")
    ax.set_ylabel("PE$_{clim}$")
    ax.set_title("Per-horizon PE (mean ± std over seeds)")
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_horizon_pe.png", dpi=200)
    plt.close()

print("\nSUCCESS! All tables and figures have been generated!")
print(f"Check the `{OUT_DIR.name}` and `{FIG_DIR.name}` folders in your project root!")


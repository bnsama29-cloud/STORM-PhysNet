# ============================================================
# NB6 — EXTRACT EPOCH LOGS TO CSV (Google Colab)
# Scans the saved .txt logs from NB1 and NB2 on Google Drive
# and compiles them into a single CSV table for your IEEE paper.
# ============================================================
import os
import glob
import pandas as pd
from pathlib import Path
import re

# -------------------- USER SETTINGS --------------------
DRIVE_ROOT = "/content/drive/MyDrive/storm_physnet"
OUT_DIR    = f"{DRIVE_ROOT}/epoch_metrics"
# -------------------------------------------------------

# 1. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

def extract_epoch_metrics():
    # Dynamically find all log files across ALL *_outputs directories (e.g., nb1_outputs, ablation_outputs)
    log_files = glob.glob(f"{DRIVE_ROOT}/*_outputs/logs/**/*.txt", recursive=True)
    
    if not log_files:
        print(f"No log files found in any {DRIVE_ROOT}/*_outputs/logs/ directory.")
        return None

    print(f"Found {len(log_files)} log files on Google Drive:")
    for lf in sorted(log_files):
        print(f"  - {Path(lf).name}")

    all_data = []

    # Regex to parse the standard trainer stdout line (handles both old and new log formats):
    # Old format: Epoch   1 | Train: 0.8172 | Val: 0.7289 | Val MSE: 0.2812
    # New format: Epoch   1 | Train: 0.8172 | Val: 0.7289 | Val MSE: 0.2812 | delay: 1.240h | LR: 1.00e-04
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
        r"Train:\s+(?P<train_loss>[\d\.\-e\+]+)\s+\|\s+"
        r"Val:\s+(?P<val_loss>[\d\.\-e\+]+)\s+\|\s+"
        r"Val MSE:\s+(?P<val_mse>[\d\.\-e\+]+)"
        r"(?:\s+\|\s+delay:\s+(?P<delay>[\d\.\-e\+]+)h)?"
        r"(?:\s+\|\s+LR:\s+(?P<lr>[\d\.\-e\+]+))?"
    )

    for log_path in log_files:
        file_name = Path(log_path).name
        # Filename format is {label}_seed{seed}.txt
        name_parts = file_name.replace(".txt", "").split("_seed")
        if len(name_parts) == 2:
            model_label, seed = name_parts[0], name_parts[1]
        else:
            model_label, seed = file_name.replace(".txt", ""), "unknown"

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            match = pattern.search(line)
            if match:
                data = match.groupdict()
                all_data.append({
                    "Model": model_label,
                    "Seed": seed,
                    "Epoch": int(data["epoch"]),
                    "Train_Loss": float(data["train_loss"]),
                    "Val_Loss": float(data["val_loss"]),
                    "Val_MSE": float(data["val_mse"]),
                    "Delay_Hours": float(data["delay"]) if data.get("delay") else 0.0,
                    "Learning_Rate": float(data["lr"]) if data.get("lr") else 0.0
                })

    if not all_data:
        print("No epoch metrics could be extracted. The regex might not be matching.")
        return None

    df = pd.DataFrame(all_data)
    
    # Sort logically
    df.sort_values(by=["Model", "Seed", "Epoch"], inplace=True)
    
    # Ensure output directory exists
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Group by Model and Seed to save separate CSVs
    csv_paths = []
    for (model, seed), group in df.groupby(["Model", "Seed"]):
        csv_name = f"{model}_seed{seed}.csv"
        csv_path = Path(OUT_DIR) / csv_name
        group.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    
    print(f"\nSuccessfully extracted {len(df)} epochs of data.")
    print(f"Generated {len(csv_paths)} separate CSV files in {OUT_DIR}.")
    
    return df, csv_paths

# 2. Run extraction and display summary
result = extract_epoch_metrics()
if result is not None:
    df, csv_paths = result
    import google.colab.data_table as dt
    from IPython.display import display
    print("\nPreview of extracted data (first 15 rows across all files):")
    display(dt.DataTable(df.head(50), include_index=False, num_rows_per_page=15))
    
    # 3. Create a ZIP file of all individual CSVs and download it automatically
    import zipfile
    from google.colab import files
    
    zip_path = "/content/drive/MyDrive/storm_physnet/epoch_metrics.zip"
    print(f"\nCompressing {len(csv_paths)} CSV files into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for csv_path in csv_paths:
            # Just store the file itself in the zip, without the long folder path
            zipf.write(csv_path, arcname=csv_path.name)
        
    print("Triggering automatic browser download...")
    files.download(zip_path)

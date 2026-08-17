
import os, zipfile, shutil
from pathlib import Path

BASE_DIR = Path(r"f:\Downloads\ieee_final_fixed\kaggle_outputs")
ORG_DIR = BASE_DIR / "Organized_Data"

# Clean up old organized dir if it exists
shutil.rmtree(ORG_DIR, ignore_errors=True)
ORG_DIR.mkdir(parents=True, exist_ok=True)

ckpt_dir = ORG_DIR / "checkpoints"
res_dir = ORG_DIR / "results"
ckpt_dir.mkdir(exist_ok=True)
res_dir.mkdir(exist_ok=True)

zips = list(BASE_DIR.glob("*.zip"))
print(f"Found {len(zips)} ZIP files. Extracting and merging...")

for z in zips:
    print(f"Extracting {z.name}...")
    if "ckpt" in z.name:
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(ckpt_dir)
    elif "results" in z.name:
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(res_dir)

print("\nSuccessfully organized all files into:")
print(f"{ORG_DIR}")
print("Go check the folder! It is neatly grouped by Model -> Seed.")


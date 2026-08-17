import os
import sys
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("STORM-PhysNet_Test")

if not REPO_DIR.exists():
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print("Pulling latest fixes...")
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml", "papermill", "ipykernel"], check=True)

os.chdir(REPO_DIR)

MY_SEEDS = [42]

override_code = f"""
import json
with open('notebooks/STORM_PhysNet_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if "DEMO_MODE = True" in src:
            new_src = src.replace("SEEDS = list(range(42, 57))", "SEEDS = {MY_SEEDS}")
            cell['source'] = [new_src]

with open('notebooks/STORM_PhysNet_Colab_RUN.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
"""
with open("override.py", "w", encoding="utf-8") as f:
    f.write(override_code)
subprocess.run([sys.executable, "override.py"], check=True)

print(f"Executing full pipeline for seeds {MY_SEEDS}...")
res = subprocess.run([sys.executable, "-m", "papermill", 
                "notebooks/STORM_PhysNet_Colab_RUN.ipynb", 
                "notebooks/STORM_PhysNet_Colab_OUT.ipynb"])

if res.returncode != 0:
    print("PAPERMILL FAILED")
    sys.exit(1)

out_dir = Path("../STORM_Results_Account_Export")
out_dir.mkdir(exist_ok=True)

if Path("checkpoints").exists():
    shutil.copytree("checkpoints", out_dir / "checkpoints", dirs_exist_ok=True)
if Path("results").exists():
    shutil.copytree("results", out_dir / "results", dirs_exist_ok=True)
if Path("figures").exists():
    shutil.copytree("figures", out_dir / "figures", dirs_exist_ok=True)
if Path("logs").exists():
    shutil.copytree("logs", out_dir / "logs", dirs_exist_ok=True)
    
shutil.copy("notebooks/STORM_PhysNet_Colab_OUT.ipynb", out_dir / "executed_notebook.ipynb")

shutil.make_archive("../STORM_results", "zip", out_dir)
print("Done!")


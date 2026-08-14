# ===== WEEK 2: SELECT ABLATION BY NAME FROM CONFIG =====
# EDIT THESE TWO LINES PER ACCOUNT:
ABLATION_NAME = "A_45min_only"   # <-- CHANGE: A_45min_only | B_arch_only | C_mild_6h | D_mod_6h
SEEDS = [42, 43, 44, 45, 46]     # <-- CHANGE PER BATCH

import os, shutil, yaml, subprocess
from pathlib import Path
import torch, sys

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE = Path("/content/drive/MyDrive/storm_physnet")
CODE_ZIP = DRIVE / "ieee_final_fixed.zip"
DATA_ZIP = DRIVE / "datasets.zip"
OUT = DRIVE / f"week2_ablation_{ABLATION_NAME}"
OUT.mkdir(parents=True, exist_ok=True)

os.chdir("/content")
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm scipy matplotlib")

WORK = Path("/content/storm_work")
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir(exist_ok=True)
os.chdir(WORK)

print("Extracting...")
shutil.copy2(CODE_ZIP, WORK / "code.zip")
subprocess.run(["unzip", "-q", "-o", "code.zip"], cwd=WORK, capture_output=True)
sys.path.insert(0, str(WORK))
import src
print(f"src ok: {src.__file__}")

for key in ["goes", "omni", "grasp"]:
    dst = Path("datasets") / key
    if not dst.exists():
        alt = DRIVE / "datasets" / key
        if alt.exists(): shutil.copytree(alt, dst)
        else:
            shutil.unpack_archive(str(DATA_ZIP), WORK / "_data")
            hits = list(Path("_data").glob(f"**/{key}"))
            if hits: shutil.copytree(hits[0], dst)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")

# ---- LOAD ABLATION FROM CONFIG PRESETS ----
base_cfg = yaml.safe_load(open("configs/config_week2.yaml"))
presets = base_cfg["loss"]["ablation_presets"]
if ABLATION_NAME not in presets:
    raise ValueError(f"Unknown ablation: {ABLATION_NAME}. Available: {list(presets.keys())}")
ablation_cfg = presets[ABLATION_NAME]
print(f"Running ablation: {ABLATION_NAME} - {ablation_cfg['description']}")
print(f"  physics_horizon_scale = {ablation_cfg['physics_horizon_scale']}")

# ---- TRAIN SEEDS ----
for seed in SEEDS:
    cfg = base_cfg.copy()
    cfg["loss"]["physics_horizon_scale"] = ablation_cfg["physics_horizon_scale"]
    cfg["training"]["seed"] = seed
    ck = OUT / "checkpoints" / f"storm_hz_{ABLATION_NAME}" / f"seed_{seed}"
    ck.mkdir(parents=True, exist_ok=True)
    cfg["training"]["checkpoint_dir"] = str(ck)
    cfg["training"]["log_dir"] = str(OUT / "logs" / f"seed_{seed}")
    yaml.safe_dump(cfg, open("configs/config_week2_run.yaml", "w"), sort_keys=False)

    cmd = (
        "python -u run_training.py --config configs/config_week2_run.yaml "
        "--model storm_physnet --no-ensemble --ablation none "
        "--gate-type bz --backbone transformer"
    )
    log = ck / "train.log"
    print("=" * 60, f"\n{ABLATION_NAME} seed {seed}")
    ret = os.system(f"{cmd} > {log} 2>&1")
    print("exit", ret)
    if log.exists():
        print("\n".join(log.read_text(errors="ignore").splitlines()[-15:]))
    arts = list(ck.glob("*_best.pt"))
    print("artifacts", [a.name for a in arts])

print(f"DONE {ABLATION_NAME} seeds {SEEDS} → {OUT}")
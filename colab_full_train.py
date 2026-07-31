# ============================================================
# IEEE STORM-PhysNet Full Experimental Pipeline
# Combines all training, extraction, evaluation, GRASP transfer, and plotting.
# ============================================================



# ======================================================================
# --- PHASE: 01_colab_train_main.py ---
# ======================================================================

# ============================================================
# NB1 — MAIN multi-seed train (Google Colab / GPU)
# transformer | storm_bz | storm_cathode | storm_cathode_spec | storm_radiotrophic
# Seeds 42, 43, 44
# ============================================================
import os, glob, shutil, zipfile
from pathlib import Path
import yaml

# -------------------- USER SETTINGS --------------------
# Option A: zip already on Drive
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"   # change path
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"          # goes+omni
# Where checkpoints are saved (persist on Drive)
DRIVE_OUT = "/content/drive/MyDrive/storm_physnet/nb1_outputs"

# Option B: if you uploaded zips to /content via the Files panel, set:
# DRIVE_CODE_ZIP = "/content/ieee_final_fixed.zip"
# DRIVE_DATA_ZIP = "/content/datasets.zip"

SEEDS = [42, 43, 44]
DO_TRAIN = True
SKIP_IF_CKPT_EXISTS = True   # resume after Colab disconnect
# -------------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime → Change runtime type → GPU (T4)"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm")

# Unpack code
code_zip = Path(DRIVE_CODE_ZIP)
assert code_zip.exists(), f"Code zip not found: {code_zip}"
with zipfile.ZipFile(code_zip, "r") as z:
    z.extractall(WORK / "_code")
hits = list((WORK / "_code").rglob("run_training.py"))
assert hits, "run_training.py not inside code zip"
code_root = hits[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
shutil.copy2(code_root / "run_training.py", WORK / "run_training.py")
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()
print("code_root:", code_root)

# Unpack data (goes + omni)
dst_goes, dst_omni = WORK / "datasets" / "goes", WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    data_zip = Path(DRIVE_DATA_ZIP)
    assert data_zip.exists(), f"Data zip not found: {data_zip}"
    with zipfile.ZipFile(data_zip, "r") as z:
        z.extractall(WORK / "_data")
    # find goes/ and omni/ folders
    g = next((p for p in (WORK / "_data").rglob("goes") if p.is_dir()), None)
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    assert g and o, "datasets zip must contain goes/ and omni/ folders"
    dst_goes.parent.mkdir(parents=True, exist_ok=True)
    if dst_goes.exists():
        shutil.rmtree(dst_goes)
    if dst_omni.exists():
        shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)
print("GOES files:", len(list(dst_goes.glob('*.cdf'))), "OMNI:", list(dst_omni.glob('*'))[:5])

Path("checkpoints").mkdir(exist_ok=True)
Path("logs/nb1").mkdir(parents=True, exist_ok=True)
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

JOBS = [
    # label, model, ablation, gate, spectral, backbone
    ("transformer", "transformer", "none", "bz", False, "transformer"),
    ("storm_bz", "storm_physnet", "none", "bz", False, "transformer"),
    ("storm_cathode", "storm_physnet", "none", "cathode_anode", False, "transformer"),
    ("storm_cathode_spec", "storm_physnet", "none", "cathode_anode", True, "transformer"),
    ("storm_radiotrophic", "storm_physnet", "none", "radiotrophic", False, "transformer"),
]

def write_cfg(seed, gate, spectral, backbone, ckpt_dir):
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg["data"]["goes_cdf_dir"] = "datasets/goes"
    cfg["data"]["wind_cdf_dir"] = "datasets/omni"
    cfg["data"]["batch_size"] = int(cfg.get("data", {}).get("batch_size") or 64)
    cfg["data"]["num_workers"] = 0
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = int(cfg["training"].get("epochs", 40))
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = "logs/nb1"
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = backbone
    cfg["model"]["gate_type"] = gate
    cfg["model"]["use_spectral_head"] = bool(spectral)
    cfg.setdefault("transfer", {})["enabled"] = False
    yaml.safe_dump(cfg, open("configs/config.yaml", "w"), sort_keys=False)

def sync_to_drive(label, seed):
    """Copy finished job to Drive so disconnect does not lose work."""
    src = Path(f"checkpoints/{label}/seed_{seed}")
    dst = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    log = Path(f"logs/nb1/{label}_seed{seed}.txt")
    if log.exists():
        dlog = Path(DRIVE_OUT) / "logs" / "nb1"
        dlog.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dlog / log.name)

def has_ckpt(label, seed):
    d = Path(f"checkpoints/{label}/seed_{seed}")
    d2 = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
    for base in (d, d2):
        if list(base.glob("*_best.pt")) or list(base.glob("*_best.zip")):
            return True
    return False

def run_train(label, seed, model, ablation, gate, spectral, backbone):
    if SKIP_IF_CKPT_EXISTS and has_ckpt(label, seed):
        print(f"SKIP {label} seed={seed} (checkpoint exists)")
        # ensure local copy for later jobs in this session
        d2 = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
        d = Path(f"checkpoints/{label}/seed_{seed}")
        if d2.exists() and not d.exists():
            shutil.copytree(d2, d)
        return

    d = Path(f"checkpoints/{label}/seed_{seed}")
    d.mkdir(parents=True, exist_ok=True)
    write_cfg(seed, gate, spectral, backbone, d)
    log = Path(f"logs/nb1/{label}_seed{seed}.txt")
    extra = ""
    if model == "storm_physnet":
        extra += f" --gate-type {gate} --backbone {backbone}"
        if spectral:
            extra += " --spectral-head"
    cmd = (
        f"python -u run_training.py --config configs/config.yaml "
        f"--model {model} --no-ensemble --ablation {ablation}{extra}"
    )
    print("\n" + "=" * 72)
    print(f"NB1 {label} seed={seed} gate={gate} spectral={spectral}")
    print(cmd)
    print("=" * 72)
    ret = os.system(f"{cmd} > {log} 2>&1")
    print(f"exit={ret}")
    if log.exists():
        print("\n".join(log.read_text(errors="ignore").splitlines()[-15:]))
    arts = list(d.glob("*_best.pt")) + list(d.glob("*_best.zip")) + list(d.glob("preprocessor.pkl"))
    print("artifacts:", [a.name for a in arts])
    if not arts:
        print("WARNING: no checkpoint for", label, seed)
    else:
        sync_to_drive(label, seed)
        print("synced ->", DRIVE_OUT)

if DO_TRAIN:
    for seed in SEEDS:
        for label, model, ablation, gate, spectral, backbone in JOBS:
            run_train(label, seed, model, ablation, gate, spectral, backbone)

print("\nNB1 COMPLETE")
print("Local:", WORK / "checkpoints")
print("Drive:", DRIVE_OUT)


# ======================================================================
# --- PHASE: 02_colab_train_optional.py ---
# ======================================================================

# ============================================================
# NB2 — OPTIONAL / EXPLORATORY train (Google Colab / GPU)
# Ablations: no_delay, no_physics
# Baselines: lstm, mlp, cnn
# storm_weight sweep (10/15/20) on cathode+spectral
# seq_len 48/96 on cathode+spectral
# hybrid backbone (SSM+transformer), seed 42
# radiotrophic + spectral combo, seed 42
# storm_bz_mag (magnetopause geometry added)
# ============================================================
import os, glob, shutil, zipfile
from pathlib import Path
import yaml

# -------------------- USER SETTINGS --------------------
# Option A: zip already on Drive
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"   # change path
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"          # goes+omni
# Where checkpoints are saved (persist on Drive)
DRIVE_OUT = "/content/drive/MyDrive/storm_physnet/nb2_outputs"

# Option B: if you uploaded zips to /content via the Files panel, set:
# DRIVE_CODE_ZIP = "/content/ieee_final_fixed.zip"
# DRIVE_DATA_ZIP = "/content/datasets.zip"

SEED = 42
DO_ABLATIONS = True
DO_BASELINES = True
DO_SWEEP = True
DO_SEQLEN = True
DO_HYBRID = True          # SSM+transformer hybrid backbone
DO_RADIO_SPEC = True      # radiotrophic + spectral combo
DO_MAG = True             # NEW: storm_bz + magnetopause geometry (Shue 1998)
SWEEP_WEIGHTS = [10.0, 15.0, 20.0]
SEQ_LENS = [48, 96]

SKIP_IF_CKPT_EXISTS = True   # resume after Colab disconnect
# -------------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime -> Change runtime type -> GPU (T4)"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm")

# -------------------- Unpack code --------------------
code_zip = Path(DRIVE_CODE_ZIP)
assert code_zip.exists(), f"Code zip not found: {code_zip}"
with zipfile.ZipFile(code_zip, "r") as z:
    z.extractall(WORK / "_code")
hits = list((WORK / "_code").rglob("run_training.py"))
assert hits, "run_training.py not inside code zip"
code_root = hits[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
shutil.copy2(code_root / "run_training.py", WORK / "run_training.py")
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()
print("code_root:", code_root)

# -------------------- Unpack data (goes + omni) --------------------
dst_goes, dst_omni = WORK / "datasets" / "goes", WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    data_zip = Path(DRIVE_DATA_ZIP)
    assert data_zip.exists(), f"Data zip not found: {data_zip}"
    with zipfile.ZipFile(data_zip, "r") as z:
        z.extractall(WORK / "_data")
    g = next((p for p in (WORK / "_data").rglob("goes") if p.is_dir()), None)
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    assert g and o, "datasets zip must contain goes/ and omni/ folders"
    dst_goes.parent.mkdir(parents=True, exist_ok=True)
    if dst_goes.exists():
        shutil.rmtree(dst_goes)
    if dst_omni.exists():
        shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)
print("GOES files:", len(list(dst_goes.glob('*.cdf'))), "OMNI:", list(dst_omni.glob('*'))[:5])

Path("checkpoints").mkdir(exist_ok=True)
Path("logs/nb2").mkdir(parents=True, exist_ok=True)
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

# label, model, ablation, gate, spectral, backbone, storm_weight, seq_len
JOBS = []

if DO_ABLATIONS:
    JOBS += [
        ("storm_no_delay", "storm_physnet", "no_delay", "bz", False, "transformer", None, None),
        ("storm_no_physics", "storm_physnet", "no_physics", "bz", False, "transformer", None, None),
    ]

if DO_BASELINES:
    JOBS += [
        ("lstm", "lstm", "none", "bz", False, "transformer", None, None),
        ("mlp", "mlp", "none", "bz", False, "transformer", None, None),
        ("cnn", "cnn", "none", "bz", False, "transformer", None, None),
    ]

if DO_SWEEP:
    for w in SWEEP_WEIGHTS:
        JOBS.append((f"sweep_sw{int(w)}", "storm_physnet", "none",
                     "cathode_anode", True, "transformer", w, None))

if DO_SEQLEN:
    for sl in SEQ_LENS:
        JOBS.append((f"seqlen_{sl}", "storm_physnet", "none",
                     "cathode_anode", True, "transformer", None, sl))

if DO_HYBRID:
    JOBS.append(("storm_hybrid", "storm_physnet", "none", "bz", False, "hybrid", None, None))

if DO_RADIO_SPEC:
    JOBS.append(("storm_radio_spec", "storm_physnet", "none", "radiotrophic", True, "transformer", None, None))


def write_cfg(seed, gate, spectral, backbone, ckpt_dir, storm_weight=None, seq_len=None):
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg["data"]["goes_cdf_dir"] = "datasets/goes"
    cfg["data"]["wind_cdf_dir"] = "datasets/omni"
    cfg["data"]["batch_size"] = int(cfg.get("data", {}).get("batch_size") or 64)
    cfg["data"]["num_workers"] = 0
    if seq_len is not None:
        cfg["data"]["sequence_length"] = int(seq_len)
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = int(cfg["training"].get("epochs", 40))
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = "logs/nb2"
    if storm_weight is not None:
        cfg["training"]["storm_weight"] = float(storm_weight)
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = backbone
    cfg["model"]["gate_type"] = gate
    cfg["model"]["use_spectral_head"] = bool(spectral)
    cfg.setdefault("transfer", {})["enabled"] = False
    yaml.safe_dump(cfg, open("configs/config.yaml", "w"), sort_keys=False)


def sync_to_drive(label, seed):
    """Copy finished job to Drive so a Colab disconnect does not lose work."""
    src = Path(f"checkpoints/{label}/seed_{seed}")
    dst = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    log = Path(f"logs/nb2/{label}_seed{seed}.txt")
    if log.exists():
        dlog = Path(DRIVE_OUT) / "logs" / "nb2"
        dlog.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dlog / log.name)


def has_ckpt(label, seed):
    d = Path(f"checkpoints/{label}/seed_{seed}")
    d2 = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
    for base in (d, d2):
        if list(base.glob("*_best.pt")) or list(base.glob("*_best.zip")):
            return True
    return False


def run_train(label, seed, model, ablation, gate, spectral, backbone,
              storm_weight=None, seq_len=None):
    if SKIP_IF_CKPT_EXISTS and has_ckpt(label, seed):
        print(f"SKIP {label} seed={seed} (checkpoint exists)")
        d2 = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
        d = Path(f"checkpoints/{label}/seed_{seed}")
        if d2.exists() and not d.exists():
            shutil.copytree(d2, d)
        return

    d = Path(f"checkpoints/{label}/seed_{seed}")
    d.mkdir(parents=True, exist_ok=True)
    write_cfg(seed, gate, spectral, backbone, d, storm_weight, seq_len)
    log = Path(f"logs/nb2/{label}_seed{seed}.txt")
    extra = ""
    if model == "storm_physnet":
        extra += f" --gate-type {gate} --backbone {backbone}"
        if spectral:
            extra += " --spectral-head"
        if "mag" in label:
            extra += " --magnetopause"
    cmd = (
        f"python -u run_training.py --config configs/config.yaml "
        f"--model {model} --no-ensemble --ablation {ablation}{extra}"
    )
    print("\n" + "=" * 72)
    print(f"NB2 {label} seed={seed} model={model} abl={ablation} gate={gate} "
          f"spec={spectral} backbone={backbone} sw={storm_weight} seq={seq_len}")
    print(cmd)
    print("=" * 72)
    ret = os.system(f"{cmd} > {log} 2>&1")
    print(f"exit={ret}")
    if log.exists():
        print("\n".join(log.read_text(errors="ignore").splitlines()[-15:]))
    arts = list(d.glob("*_best.pt")) + list(d.glob("*_best.zip")) + list(d.glob("preprocessor.pkl"))
    print("artifacts:", [a.name for a in arts])
    if not arts:
        print("WARNING: no checkpoint for", label, seed)
    else:
        sync_to_drive(label, seed)
        print("synced ->", DRIVE_OUT)


for label, model, ablation, gate, spectral, backbone, storm_weight, seq_len in JOBS:
    run_train(label, SEED, model, ablation, gate, spectral, backbone, storm_weight, seq_len)

if DO_MAG:
    # storm_bz + magnetopause: adds Shue (1998) r0/alpha/compression
    # to the encoder output before the Bz gate.
    run_train("storm_bz_mag", SEED, "storm_physnet", "none", "bz", False, "transformer")

print("\nNB2 COMPLETE — magnetopause_geometry.py is in src/model/ (see README).")
print("Local:", WORK / "checkpoints")
print("Drive:", DRIVE_OUT)


# ======================================================================
# --- PHASE: 06_colab_extract_logs.py ---
# ======================================================================

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
DRIVE_NB1_LOGS = "/content/drive/MyDrive/storm_physnet/nb1_outputs/logs/nb1"
DRIVE_NB2_LOGS = "/content/drive/MyDrive/storm_physnet/nb2_outputs/logs/nb2"
OUT_DIR        = "/content/drive/MyDrive/storm_physnet/epoch_metrics"
# -------------------------------------------------------

# 1. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

def extract_epoch_metrics():
    log_files = glob.glob(f"{DRIVE_NB1_LOGS}/*.txt") + glob.glob(f"{DRIVE_NB2_LOGS}/*.txt")
    
    if not log_files:
        print(f"No log files found in {DRIVE_NB1_LOGS} or {DRIVE_NB2_LOGS}")
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


# ======================================================================
# --- PHASE: 03_colab_eval.py ---
# ======================================================================

# ============================================================
# NB3 — FULL EVAL + IEEE TABLE + FIGURES (Google Colab)
# Evaluates ALL checkpoints from NB1 + NB2 and writes:
#   logs/full_run/ieee_table.txt  — copy-paste into LaTeX
#   logs/full_run/summary.json   — full metrics
#   plots/fig1_horizon_pe.png    — per-horizon bar chart
#   plots/fig2_all_vs_storm.png  — PE-all vs PE-storm scatter
#   plots/fig3_storm_weight_sweep.png
#   plots/fig4_seqlen_ablation.png
# ============================================================
import os, glob, shutil, json, pickle, zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- USER SETTINGS --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_NB1_OUT  = "/content/drive/MyDrive/storm_physnet/nb1_outputs"
DRIVE_NB2_OUT  = "/content/drive/MyDrive/storm_physnet/nb2_outputs"
DRIVE_NB3_OUT  = "/content/drive/MyDrive/storm_physnet/nb3_outputs"
# -------------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

assert torch.cuda.is_available(), "Enable GPU: Runtime -> Change runtime type -> GPU"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib")

# -------------------- Unpack code --------------------
code_zip = Path(DRIVE_CODE_ZIP)
assert code_zip.exists(), f"Code zip not found: {code_zip}"
with zipfile.ZipFile(code_zip, "r") as z:
    z.extractall(WORK / "_code")
hits = list((WORK / "_code").rglob("run_training.py"))
assert hits, "run_training.py not inside code zip"
code_root = hits[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# -------------------- Unpack data --------------------
dst_goes = WORK / "datasets" / "goes"
dst_omni = WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    with zipfile.ZipFile(Path(DRIVE_DATA_ZIP), "r") as z:
        z.extractall(WORK / "_data")
    g = next((p for p in (WORK / "_data").rglob("goes") if p.is_dir()), None)
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    assert g and o, "datasets zip must contain goes/ and omni/ folders"
    if dst_goes.exists(): shutil.rmtree(dst_goes)
    if dst_omni.exists(): shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)

# -------------------- Restore checkpoints from NB1 + NB2 Drive outputs ----
Path("checkpoints").mkdir(exist_ok=True)
for drive_out in [DRIVE_NB1_OUT, DRIVE_NB2_OUT]:
    for p in Path(drive_out).rglob("seed_*"):
        if not p.is_dir(): continue
        if not (any(p.glob("*_best.pt")) or any(p.glob("*_best.zip"))): continue
        label = p.parent.name
        dest = WORK / "checkpoints" / label / p.name
        if not dest.exists():
            shutil.copytree(p, dest)
            print(f"  restored {label}/{p.name}")

Path("logs/full_run").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# -------------------- Constants --------------------
SEEDS_MAIN = [42, 43, 44]
SEED_OPT = 42
SWEEP_WEIGHTS = [10, 15, 20]
SEQ_LENS = [48, 96]
HORIZON_NAMES = ["45min", "6h", "12h"]
HIGH_FLUX_PERCENTILE = 90

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import StandardLSTM, StandardMLP, StandardCNN, VanillaTransformer
from src.model.storm_physnet import STORMPhysNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goes = read_goes_directory("datasets/goes")
wind = read_wind_directory("datasets/omni")
raw = goes.join(wind, how="inner")
print("[Data]", raw.shape)

pre_path = next(Path("checkpoints").rglob("preprocessor.pkl"), None) or \
           next(Path(DRIVE_NB1_OUT).rglob("preprocessor.pkl"), None)
try:
    if pre_path:
        pre = pickle.load(open(pre_path, "rb"))
    else:
        pre = Preprocessor()
    train_df, val_df, test_df = pre.fit_transform(raw)
except Exception as e:
    print(f"Warning: Preprocessor load failed ({e}). Re-fitting from scratch.")
    pre = Preprocessor()
    train_df, val_df, test_df = pre.fit_transform(raw)
print(f"split train={len(train_df)} val={len(val_df)} test={len(test_df)}")

_loader_cache = {}
def get_test_loader(seq_len):
    if seq_len not in _loader_cache:
        _, _, tl = make_dataloaders(train_df, val_df, test_df, seq_len=seq_len,
                                    batch_size=64, storm_weight=10.0, num_workers=0)
        _loader_cache[seq_len] = tl
    return _loader_cache[seq_len]

def pe(yt, yp, yb):
    mse_p, mse_b = mean_squared_error(yt, yp), mean_squared_error(yt, yb)
    return 0.0 if mse_b == 0 else float(1.0 - mse_p / mse_b)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
def bias(yt, yp): return float(np.mean(yp - yt))

def find_ckpt(d):
    d = Path(d)
    if not d.exists(): return None
    cands = sorted(d.glob("*_best.pt")) + sorted(d.glob("*_best.zip"))
    return cands[0] if cands else None

def build_model(label, n_sw, seq_len):
    if label == "lstm": return StandardLSTM(n_sw_features=n_sw, seq_len=seq_len, n_horizons=3)
    if label == "mlp":  return StandardMLP(n_sw_features=n_sw, seq_len=seq_len, n_horizons=3)
    if label == "cnn":  return StandardCNN(n_sw_features=n_sw, seq_len=seq_len, n_horizons=3)
    if label == "transformer": return VanillaTransformer(n_sw_features=n_sw, seq_len=seq_len, n_horizons=3)
    gate = "bz"
    if "cathode" in label or "sweep" in label or "seqlen" in label: gate = "cathode_anode"
    if "radio" in label: gate = "radiotrophic"
    spec     = ("spec" in label) or ("sweep" in label) or ("seqlen" in label)
    backbone = "hybrid" if "hybrid" in label else "transformer"
    abl      = "none"
    if "no_delay"   in label: abl = "no_delay"
    if "no_physics" in label: abl = "no_physics"
    use_mag  = "mag" in label
    return STORMPhysNet(
        n_sw_features=n_sw, seq_len=seq_len, d_model=128, n_heads=4,
        n_transformer_layers=2, n_ssm_layers=2, d_state=64, d_ff=256,
        hidden_dim=64, n_horizons=3, dropout=0.1, ablation=abl,
        backbone=backbone, gate_type=gate, use_spectral_head=spec,
        use_magnetopause=use_mag,
    )

def load_state(model, path):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    res = model.load_state_dict(state, strict=False)
    n_bad = len(res.missing_keys) + len(res.unexpected_keys)
    if n_bad / max(len(model.state_dict()), 1) > 0.05:
        print(f"  [WARNING] {path.name}: {n_bad} key mismatches")
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues, bases, storms = [], [], [], []
    for batch in loader:
        x_sw   = torch.nan_to_num(batch["x_sw"].to(device), nan=0.0)
        x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
        y_p    = batch["y_persist"].to(device)
        try:    out = model(x_sw, x_flux, y_p)
        except TypeError: out = model(x_sw, x_flux)
        yp = out["flux_pred"] if isinstance(out, dict) else out
        preds.append(yp.cpu().numpy())
        trues.append(batch["y_flux"].numpy())
        bases.append(batch["y_persist"].numpy())
        storms.append(batch["storm_flag"].numpy().ravel())
    yt, yp, yb = map(np.concatenate, (trues, preds, bases))
    st = np.concatenate(storms, 0).astype(bool).reshape(-1)
    if st.shape[0] != yt.shape[0]: st = np.zeros(yt.shape[0], dtype=bool)
    return yt, yp, yb, st

def metrics_block(yt, yp, yb, st):
    out = {}
    for i, h in enumerate(HORIZON_NAMES):
        out[f"pe_all_{h}"]   = pe(yt[:, i], yp[:, i], yb[:, i])
        out[f"rmse_all_{h}"] = rmse(yt[:, i], yp[:, i])
        out[f"corr_all_{h}"] = corr(yt[:, i], yp[:, i])
        out[f"bias_all_{h}"] = bias(yt[:, i], yp[:, i])
        if st.any():
            out[f"pe_storm_{h}"]   = pe(yt[st, i], yp[st, i], yb[st, i])
            out[f"rmse_storm_{h}"] = rmse(yt[st, i], yp[st, i])
        else:
            out[f"pe_storm_{h}"] = out[f"rmse_storm_{h}"] = float("nan")
    h = 1
    thr = np.percentile(yt[:, h], HIGH_FLUX_PERCENTILE)
    m = yt[:, h] >= thr
    if m.any():
        out["pe_highflux_6h"]   = pe(yt[m, h], yp[m, h], yb[m, h])
        out["rmse_highflux_6h"] = rmse(yt[m, h], yp[m, h])
        out["n_highflux"]       = int(m.sum())
    else:
        out["pe_highflux_6h"] = out["rmse_highflux_6h"] = float("nan")
        out["n_highflux"] = 0
    return out

def eval_one(label, seed, seq_len=72):
    d    = Path(f"checkpoints/{label}/seed_{seed}")
    ckpt = find_ckpt(d)
    if ckpt is None:
        print(f"{label:<24} seed={seed} MISSING")
        return None
    try:
        loader = get_test_loader(seq_len)
        model  = load_state(build_model(label, loader.dataset.n_sw_features, seq_len).to(device), ckpt)
        yt, yp, yb, st = predict(model, loader)
        row = metrics_block(yt, yp, yb, st)
        row.update({"seed": seed, "label": label, "seq_len": seq_len, "ckpt": str(ckpt)})
        return row, (yt, yp, yb, st)
    except Exception as e:
        print(f"{label:<24} seed={seed} ERROR {type(e).__name__}: {e}")
        return None

MAIN_MULTI  = ["transformer", "storm_bz", "storm_cathode", "storm_cathode_spec", "storm_radiotrophic"]
MAIN_SINGLE = ["lstm", "mlp", "cnn", "storm_no_delay", "storm_no_physics",
               "storm_hybrid", "storm_radio_spec", "storm_bz_mag"]
SWEEP_LABELS  = [f"sweep_sw{w}" for w in SWEEP_WEIGHTS]
SEQLEN_LABELS = [f"seqlen_{sl}" for sl in SEQ_LENS]

results    = {lab: [] for lab in MAIN_MULTI + MAIN_SINGLE + SWEEP_LABELS + SEQLEN_LABELS}
pred_cache = {}
summary    = {"main_jobs": {}, "single_seed": {}, "sweep": {}, "seqlen": {}, "ensembles": {}}

print(f"\n{'='*72}")
print(f"{'label':<24} seed   PE45   PE6    PE12   PEst6  PEhi")
print(f"{'='*72}")

for seed in SEEDS_MAIN:
    for label in MAIN_MULTI:
        out = eval_one(label, seed, 72)
        if not out: continue
        row, cache = out
        results[label].append(row)
        pred_cache[(label, seed)] = cache
        print(f"{label:<24} {seed:>4}  {row['pe_all_45min']:5.3f} {row['pe_all_6h']:5.3f} "
              f"{row['pe_all_12h']:5.3f} {row['pe_storm_6h']:5.3f} {row['pe_highflux_6h']:5.3f}")

for label in MAIN_SINGLE:
    out = eval_one(label, SEED_OPT, 72)
    if not out: continue
    row, cache = out
    results[label].append(row)
    pred_cache[(label, SEED_OPT)] = cache
    summary["single_seed"][label] = row
    print(f"{label:<24} {SEED_OPT:>4}  {row['pe_all_45min']:5.3f} {row['pe_all_6h']:5.3f} "
          f"{row['pe_all_12h']:5.3f} {row['pe_storm_6h']:5.3f} {row['pe_highflux_6h']:5.3f}")

def arr(rows, k): return np.array([r[k] for r in rows], dtype=float)

print("\nMEAN +/- STD")
for label in MAIN_MULTI:
    rows = results[label]
    if not rows: continue
    entry = {
        "n": len(rows),
        "pe_all_6h_mean":      float(arr(rows, "pe_all_6h").mean()),
        "pe_all_6h_std":       float(arr(rows, "pe_all_6h").std()),
        "pe_storm_6h_mean":    float(np.nanmean(arr(rows, "pe_storm_6h"))),
        "pe_storm_6h_std":     float(np.nanstd(arr(rows, "pe_storm_6h"))),
        "pe_highflux_6h_mean": float(np.nanmean(arr(rows, "pe_highflux_6h"))),
        "corr_all_6h_mean":    float(np.nanmean(arr(rows, "corr_all_6h"))),
        "pe_all_45min_mean":   float(arr(rows, "pe_all_45min").mean()),
        "pe_all_12h_mean":     float(arr(rows, "pe_all_12h").mean()),
        "rmse_all_6h_mean":    float(arr(rows, "rmse_all_6h").mean()),
        "per_seed": rows,
    }
    summary["main_jobs"][label] = entry
    print(f"{label:<24} PE6 {entry['pe_all_6h_mean']:.4f}+/-{entry['pe_all_6h_std']:.4f} "
          f"storm {entry['pe_storm_6h_mean']:.4f}+/-{entry['pe_storm_6h_std']:.4f} "
          f"highflux {entry['pe_highflux_6h_mean']:.4f}")

for lab in SWEEP_LABELS:
    out = eval_one(lab, SEED_OPT, 72)
    if out:
        row, cache = out
        results[lab].append(row); pred_cache[(lab, SEED_OPT)] = cache
        summary["sweep"][lab] = row
        print(f"{lab:<24} PE6 {row['pe_all_6h']:.4f} PEstorm {row['pe_storm_6h']:.4f}")

for sl, lab in zip(SEQ_LENS, SEQLEN_LABELS):
    out = eval_one(lab, SEED_OPT, seq_len=sl)
    if out:
        row, cache = out
        results[lab].append(row); pred_cache[(lab, SEED_OPT)] = cache
        summary["seqlen"][lab] = row

def ensemble_multiseed(labels):
    all_yps = []; yt_ref = yb_ref = st_ref = None
    for label in labels:
        seed_preds = []
        for seed in SEEDS_MAIN:
            cache = pred_cache.get((label, seed))
            if cache is not None:
                if yt_ref is None: yt_ref, yb_ref, st_ref = cache[0], cache[2], cache[3]
                seed_preds.append(cache[1])
        if not seed_preds:
            cache = pred_cache.get((label, SEED_OPT))
            if cache is not None:
                if yt_ref is None: yt_ref, yb_ref, st_ref = cache[0], cache[2], cache[3]
                seed_preds.append(cache[1])
        if seed_preds:
            all_yps.append(np.mean(seed_preds, axis=0))
    if not all_yps or yt_ref is None: return None
    return metrics_block(yt_ref, np.mean(all_yps, axis=0), yb_ref, st_ref)

print("\nENSEMBLES (multi-seed, multi-architecture output average)")
for name, labs in [
    ("tf+bz",              ["transformer", "storm_bz"]),
    ("tf+cathode",         ["transformer", "storm_cathode"]),
    ("tf+bz+cathode",      ["transformer", "storm_bz", "storm_cathode"]),
    ("tf+cathode+spec",    ["transformer", "storm_cathode", "storm_cathode_spec"]),
    ("tf+bz+cathode+spec", ["transformer", "storm_bz", "storm_cathode", "storm_cathode_spec"]),
    ("tf+bz+mag",          ["transformer", "storm_bz", "storm_bz_mag"]),
]:
    e = ensemble_multiseed(labs)
    if e:
        summary["ensembles"][name] = e
        print(f"  {name:<28} PE6={e['pe_all_6h']:.4f}  PEstorm={e['pe_storm_6h']:.4f}  PEhi={e['pe_highflux_6h']:.4f}")
    else:
        print(f"  {name:<28} SKIPPED (missing predictions)")

# ── Figures ────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 10})
labels_ok = [l for l in MAIN_MULTI if summary["main_jobs"].get(l)]
if labels_ok:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(HORIZON_NAMES)); w = 0.8 / len(labels_ok)
    for i, lab in enumerate(labels_ok):
        rows = results[lab]
        vals = [np.mean([r[f"pe_all_{h}"] for r in rows]) for h in HORIZON_NAMES]
        errs = [np.std( [r[f"pe_all_{h}"] for r in rows]) for h in HORIZON_NAMES]
        ax.bar(x + (i - len(labels_ok)/2)*w + w/2, vals, w, label=lab, yerr=errs, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(["45 min", "6 h", "12 h"])
    ax.set_ylabel("PE"); ax.set_title("Per-Horizon PE — STORM-PhysNet (mean ±std, 3 seeds)")
    ax.legend(fontsize=7, ncol=2, frameon=False); ax.axhline(0, color="k", lw=0.8, ls="--")
    fig.tight_layout(); fig.savefig("plots/fig1_horizon_pe.png"); plt.close()
    print("Saved plots/fig1_horizon_pe.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab in labels_ok:
        e = summary["main_jobs"][lab]
        ax.errorbar(e["pe_all_6h_mean"], e["pe_storm_6h_mean"],
                    xerr=e["pe_all_6h_std"], yerr=e["pe_storm_6h_std"], fmt="o", capsize=3, label=lab, markersize=7)
    ax.set_xlabel("PE all — 6 h"); ax.set_ylabel("PE storm — 6 h")
    ax.set_title("All-sample vs Storm PE (3-seed mean ±std)"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("plots/fig2_all_vs_storm.png"); plt.close()
    print("Saved plots/fig2_all_vs_storm.png")

if summary["sweep"]:
    sw_labels = sorted(summary["sweep"].keys())
    sw_vals   = [int(k.replace("sweep_sw","")) for k in sw_labels]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sw_vals, [summary["sweep"][k]["pe_all_6h"] for k in sw_labels],   "o-", label="PE all")
    ax.plot(sw_vals, [summary["sweep"][k]["pe_storm_6h"] for k in sw_labels], "s--", label="PE storm")
    ax.set_xlabel("storm_weight"); ax.set_ylabel("PE")
    ax.set_title("storm_weight Sweep — Cathode+Spectral (seed 42)")
    ax.legend(frameon=False); ax.set_xticks(sw_vals)
    fig.tight_layout(); fig.savefig("plots/fig3_storm_weight_sweep.png"); plt.close()
    print("Saved plots/fig3_storm_weight_sweep.png")

# ── IEEE Table ─────────────────────────────────────────────────────────────
Path("logs/full_run/summary.json").write_text(json.dumps(summary, indent=2))
sep = "-" * 100
lines = [
    "IEEE Table — STORM-PhysNet GEO Electron Flux Forecasting",
    "6 h forecast horizon | PE = Prediction Efficiency | hi = top-10% flux subset",
    sep,
    f"{'Model':<24} {'n':>2}  {'PE_all':>7} {chr(177)+'std':>6}  {'PE_storm':>8} {chr(177)+'std':>6}  {'PE_hi':>6}  {'RMSE':>6}  {'PE_45m':>7}  {'PE_12h':>7}",
    sep, "--- Multi-seed main models (n=3 seeds) ---",
]
for lab, e in summary["main_jobs"].items():
    lines.append(
        f"{lab:<24} {e['n']:>2}  {e['pe_all_6h_mean']:7.4f} {e['pe_all_6h_std']:6.4f}  "
        f"{e['pe_storm_6h_mean']:8.4f} {e['pe_storm_6h_std']:6.4f}  "
        f"{e['pe_highflux_6h_mean']:6.4f}  {e['rmse_all_6h_mean']:6.4f}  "
        f"{e['pe_all_45min_mean']:7.4f}  {e['pe_all_12h_mean']:7.4f}"
    )
lines += [sep, "--- Single-seed baselines & ablations (seed 42) ---"]
for lab in MAIN_SINGLE:
    if lab in summary["single_seed"]:
        row = summary["single_seed"][lab]
        lines.append(
            f"{lab:<24}  1  {row['pe_all_6h']:7.4f} {'—':>6}  "
            f"{row['pe_storm_6h']:8.4f} {'—':>6}  "
            f"{row['pe_highflux_6h']:6.4f}  {row['rmse_all_6h']:6.4f}  "
            f"{row['pe_all_45min']:7.4f}  {row['pe_all_12h']:7.4f}"
        )
    else:
        lines.append(
            f"{lab:<24}  1  {'TBD':>7} {'—':>6}  "
            f"{'TBD':>8} {'—':>6}  "
            f"{'TBD':>6}  {'TBD':>6}  "
            f"{'TBD':>7}  {'TBD':>7}"
        )
lines += [sep, "--- Ensembles (multi-seed, multi-architecture output average) ---"]
for name, e in summary["ensembles"].items():
    lines.append(
        f"{name:<24}  —  {e['pe_all_6h']:7.4f} {'—':>6}  "
        f"{e['pe_storm_6h']:8.4f} {'—':>6}  "
        f"{e['pe_highflux_6h']:6.4f}  {e['rmse_all_6h']:6.4f}  "
        f"{e['pe_all_45min']:7.4f}  {e['pe_all_12h']:7.4f}"
    )
lines.append(sep)
table_text = "\n".join(lines)
Path("logs/full_run/ieee_table.txt").write_text(table_text)
print(table_text)

# Sync everything to Drive
Path(DRIVE_NB3_OUT).mkdir(parents=True, exist_ok=True)
shutil.copytree("plots",         Path(DRIVE_NB3_OUT) / "plots",  dirs_exist_ok=True)
shutil.copytree("logs/full_run", Path(DRIVE_NB3_OUT) / "logs",   dirs_exist_ok=True)
print(f"\nNB3 COMPLETE — results synced to {DRIVE_NB3_OUT}")
print("Files: ieee_table.txt | summary.json | plots/fig*.png")


# ======================================================================
# --- PHASE: 04_colab_transfer_grasp.py ---
# ======================================================================

# ============================================================
# NB4+5 — GRASP TRANSFER LEARNING + STORM TIME-SERIES PLOT
#          (Google Colab / T4 GPU)
# ============================================================
# Experiments covered (IEEE must-have set + interpretability):
#
# A. Zero-shot:   GOES storm_bz evaluated on GRASP with NO fine-tune
# B. Zero-shot:   GOES transformer evaluated on GRASP with NO fine-tune
# C. Frozen TL:   storm_bz — freeze encoder, train heads only (~5 min)
# D. Full TL:     storm_bz — fine-tune all weights                (~15 min)
# E. Scratch:     train storm_bz from scratch on GRASP alone      (~20 min)
# F. Few-shot:    frozen TL with 10% / 50% / 100% of GRASP data  (~10 min)
# G. Horizon table on GRASP (45m / 6h / 12h)
# H. Interpretability:
#    - Learned propagation delay τ histogram across all GOES seeds
#    - Bz gate activation during storm vs quiet windows
# I. Storm time-series plot (NB5 content — True vs Pred on GOES test set)
#
# Outputs (all synced to Drive):
#   nb4_outputs/
#     grasp_zero_shot/   — zero-shot checkpoints (symbolic)
#     grasp_frozen_tl/   — frozen-encoder fine-tuned checkpoint
#     grasp_full_tl/     — full fine-tuned checkpoint
#     grasp_scratch/     — scratch-trained checkpoint
#     plots/
#       fig_grasp_domain_gap.png      — zero-shot vs fine-tuned table bar chart
#       fig_grasp_horizon.png         — per-horizon PE on GRASP
#       fig_grasp_fewshot.png         — data-efficiency curve (10/50/100%)
#       fig_interp_delay_hist.png     — learned delay τ histogram
#       fig_interp_gate_activation.png— gate activation storm vs quiet
#       fig_timeseries_storm.png      — True vs Pred during major storm (GOES)
#       fig_timeseries_storm_6h.png   — clean 6h panel for paper
#     grasp_table.txt                 — copy-paste LaTeX table
#     summary_grasp.json              — all GRASP metrics
# ============================================================
import os, glob, shutil, pickle, zipfile, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# ==================== USER SETTINGS ====================
DRIVE_CODE_ZIP  = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP  = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_GRASP_ZIP = "/content/drive/MyDrive/storm_physnet/grasp.zip"
DRIVE_NB1_OUT   = "/content/drive/MyDrive/storm_physnet/nb1_outputs"
DRIVE_NB4_OUT   = "/content/drive/MyDrive/storm_physnet/nb4_outputs"

SEEDS_MAIN  = [42, 43, 44]
SEED        = 42
SEQ_LEN     = 72
GRASP_EPOCHS_FROZEN = 25   # frozen encoder (fast)
GRASP_EPOCHS_FULL   = 30   # full fine-tune
GRASP_EPOCHS_SCRATCH= 40   # train from scratch
GRASP_LR_FROZEN     = 2e-4
GRASP_LR_FULL       = 5e-5
GRASP_LR_SCRATCH    = 1e-4
FEW_SHOT_FRACS = [0.1, 0.5, 1.0]   # 10% / 50% / 100% of GRASP train
HIGH_FLUX_PERCENTILE = 90
# =======================================================

# ── Setup ──────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime → Change runtime type → GPU"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib")

# ── Unpack code ──────────────────────────────────────────────────────────────
code_zip = Path(DRIVE_CODE_ZIP)
assert code_zip.exists(), f"Code zip not found: {code_zip}"
with zipfile.ZipFile(code_zip, "r") as z:
    z.extractall(WORK / "_code")
hits = list((WORK / "_code").rglob("run_training.py"))
assert hits, "run_training.py not inside code zip"
code_root = hits[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
shutil.copy2(code_root / "run_training.py", WORK / "run_training.py")
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# ── Unpack GOES + OMNI ────────────────────────────────────────────────────
dst_goes = WORK / "datasets" / "goes"
dst_omni = WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    with zipfile.ZipFile(Path(DRIVE_DATA_ZIP), "r") as z:
        z.extractall(WORK / "_data")
    g = next((p for p in (WORK / "_data").rglob("goes") if p.is_dir()), None)
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    assert g and o
    dst_goes.parent.mkdir(parents=True, exist_ok=True)
    if dst_goes.exists(): shutil.rmtree(dst_goes)
    if dst_omni.exists(): shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)

# ── Unpack GRASP ─────────────────────────────────────────────────────────
dst_grasp = WORK / "datasets" / "grasp"
if not dst_grasp.exists():
    grasp_zip = Path(DRIVE_GRASP_ZIP)
    assert grasp_zip.exists(), f"GRASP zip not found. Upload grasp.zip to {DRIVE_GRASP_ZIP}"
    with zipfile.ZipFile(grasp_zip, "r") as z:
        z.extractall(WORK / "_grasp")
    g = next((p for p in (WORK / "_grasp").rglob("grasp") if p.is_dir()), None) or (WORK / "_grasp")
    shutil.copytree(g, dst_grasp)
    print(f"GRASP: {len(list(dst_grasp.rglob('*.txt')))} .txt files")

# ── Restore all NB1 checkpoints ──────────────────────────────────────────
Path("checkpoints").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("logs/nb4").mkdir(parents=True, exist_ok=True)
Path(DRIVE_NB4_OUT).mkdir(parents=True, exist_ok=True)

for p in Path(DRIVE_NB1_OUT).rglob("seed_*"):
    if not p.is_dir(): continue
    if not (any(p.glob("*_best.pt")) or any(p.glob("*_best.zip"))): continue
    label = p.parent.name
    dest_p = WORK / "checkpoints" / label / p.name
    if not dest_p.exists():
        shutil.copytree(p, dest_p)
        print(f"  restored {label}/{p.name}")

# ── Shared imports ────────────────────────────────────────────────────────
from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import VanillaTransformer
from src.model.storm_physnet import STORMPhysNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build GOES test loader (shared for interpretability + storm plot) ─────
goes = read_goes_directory("datasets/goes")
wind = read_wind_directory("datasets/omni")
raw  = goes.join(wind, how="inner")

pre_path = next(Path("checkpoints").rglob("preprocessor.pkl"), None)
try:
    if pre_path:
        pre = pickle.load(open(pre_path, "rb"))
    else:
        pre = Preprocessor()
    train_df, val_df, test_df = pre.fit_transform(raw)
except Exception as e:
    print(f"Warning: Preprocessor load failed ({e}). Re-fitting from scratch.")
    pre = Preprocessor()
    train_df, val_df, test_df = pre.fit_transform(raw)

_, _, goes_test_loader = make_dataloaders(
    train_df, val_df, test_df,
    seq_len=SEQ_LEN, batch_size=128, storm_weight=10.0, num_workers=0
)
n_sw = goes_test_loader.dataset.n_sw_features
print(f"GOES test set: {len(goes_test_loader.dataset)} windows | n_sw={n_sw}")

# ── Helper functions ──────────────────────────────────────────────────────
def pe(yt, yp, yb):
    mse_p = mean_squared_error(yt, yp)
    mse_b = mean_squared_error(yt, yb)
    return 0.0 if mse_b == 0 else float(1.0 - mse_p / mse_b)
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def find_ckpt(d):
    d = Path(d)
    if not d.exists(): return None
    cands = sorted(d.glob("*_best.pt")) + sorted(d.glob("*_best.zip"))
    return cands[0] if cands else None

def build_storm_bz(n_sw_feat, use_mag=False):
    return STORMPhysNet(
        n_sw_features=n_sw_feat, seq_len=SEQ_LEN, d_model=128, n_heads=4,
        n_transformer_layers=2, n_ssm_layers=2, d_state=64, d_ff=256,
        hidden_dim=64, n_horizons=3, dropout=0.1, ablation="none",
        backbone="transformer", gate_type="bz", use_spectral_head=False,
        use_magnetopause=use_mag,
    )

def build_transformer(n_sw_feat):
    return VanillaTransformer(n_sw_features=n_sw_feat, seq_len=SEQ_LEN, n_horizons=3)

def load_ckpt(model, path):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Filter out keys with shape mismatches (e.g., input layers changing from 16 to 2 SW features)
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            print(f"    [Transfer] Dropping {k} (pretrained {v.shape} != new {model_state[k].shape})")
        else:
            filtered_state[k] = v
            
    model.load_state_dict(filtered_state, strict=False)
    return model

@torch.no_grad()
def predict(model, loader):
    model.eval().to(device)
    preds, trues, bases, storms = [], [], [], []
    for batch in loader:
        x_sw   = torch.nan_to_num(batch["x_sw"].to(device),   nan=0.0)
        x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
        y_p    = batch["y_persist"].to(device)
        try:    out = model(x_sw, x_flux, y_p)
        except TypeError: out = model(x_sw, x_flux)
        yp = out["flux_pred"] if isinstance(out, dict) else out
        preds.append(yp.cpu().numpy())
        trues.append(batch["y_flux"].numpy())
        bases.append(batch["y_persist"].numpy())
        storms.append(batch.get("storm_flag", torch.zeros(x_sw.shape[0])).numpy().ravel())
    yt, yp, yb = map(np.concatenate, (trues, preds, bases))
    st = np.concatenate(storms).astype(bool)
    if st.shape[0] != yt.shape[0]: st = np.zeros(yt.shape[0], dtype=bool)
    return yt, yp, yb, st

HORIZON_NAMES = ["45min", "6h", "12h"]
def horizon_metrics(yt, yp, yb, st, label=""):
    thr = np.percentile(yt[:, 1], HIGH_FLUX_PERCENTILE)
    hi  = yt[:, 1] >= thr
    row = {"label": label}
    for i, h in enumerate(HORIZON_NAMES):
        row[f"pe_{h}"]   = pe(yt[:, i], yp[:, i], yb[:, i])
        row[f"rmse_{h}"] = rmse(yt[:, i], yp[:, i])
    row["pe_storm_6h"]   = pe(yt[st, 1], yp[st, 1], yb[st, 1]) if st.any() else float("nan")
    row["pe_highflux_6h"]= pe(yt[hi, 1], yp[hi, 1], yb[hi, 1]) if hi.any() else float("nan")
    row["n_storm_windows"] = int(st.sum())
    return row

def fmt(v):
    """Format a metric value for printing — shows nan as 'N/A'."""
    return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"

# ──────────────────────────────────────────────────────────────────────────
# H. INTERPRETABILITY (GOES-based, no GRASP needed)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("H. INTERPRETABILITY")
print("="*72)

# H1. Learned delay τ histogram across seeds ──────────────────────────────
tau_vals = {}
for label in ["transformer", "storm_bz", "storm_cathode", "storm_cathode_spec", "storm_radiotrophic"]:
    taus = []
    for seed in SEEDS_MAIN:
        ckpt = find_ckpt(f"checkpoints/{label}/seed_{seed}")
        if ckpt is None: continue
        try:
            model = build_storm_bz(n_sw) if label != "transformer" else build_transformer(n_sw)
            model = load_ckpt(model.to(device), ckpt)
            delay = getattr(model, "prop_delay", None)
            if delay is not None:
                # The module uses tau_logit_bias (a logit) — convert to hours
                logit = getattr(delay, "tau_logit_bias", None)
                if logit is not None:
                    frac = torch.sigmoid(logit).item()
                    tau_h = delay.tau_min + frac * (delay.tau_max - delay.tau_min)
                    taus.append(tau_h)
                else:
                    # fallback: try .tau directly
                    tau_h = delay.tau.item()
                    taus.append(tau_h)
        except Exception as e:
            print(f"  tau skip {label}/{seed}: {e}")
    if taus:
        tau_vals[label] = taus
        print(f"  {label:<28} τ = {np.mean(taus):.3f} ± {np.std(taus):.3f} h  (seeds={taus})")

if tau_vals:
    fig, ax = plt.subplots(figsize=(7, 4))
    all_taus = [t for ts in tau_vals.values() for t in ts]
    ax.hist(all_taus, bins=15, color="#E91E63", edgecolor="white", alpha=0.8)
    for i, (lab, taus) in enumerate(tau_vals.items()):
        ax.axvline(np.mean(taus), ls="--", lw=1.2, label=f"{lab} μ={np.mean(taus):.2f}h")
    ax.set_xlabel("Learned Propagation Delay τ (hours)")
    ax.set_ylabel("Count (across seeds)")
    ax.set_title("Learned Solar Wind–Magnetosphere Delay (all STORM seeds)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig("plots/fig_interp_delay_hist.png", dpi=300)
    plt.close()
    print("Saved plots/fig_interp_delay_hist.png")

# H2. Bz Gate activation: storm vs quiet ─────────────────────────────────
# We extract the gate output (sigmoid of the Bz projection) during
# storm and quiet windows on the GOES test set.
gate_storm_vals, gate_quiet_vals = [], []
ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
if ckpt:
    try:
        model = build_storm_bz(n_sw).to(device)
        model = load_ckpt(model, ckpt)
        model.eval()

        hooks = []
        gate_outputs = []
        def hook_fn(module, inp, out):
            gate_outputs.append(out.detach().cpu())

        # Find the gate module
        for name, mod in model.named_modules():
            if "bz_gate" in name.lower() or "gate" in name.lower():
                if isinstance(mod, nn.Linear):
                    hooks.append(mod.register_forward_hook(hook_fn))
                    break

        with torch.no_grad():
            for batch in goes_test_loader:
                x_sw   = torch.nan_to_num(batch["x_sw"].to(device),   nan=0.0)
                x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
                y_p    = batch["y_persist"].to(device)
                gate_outputs.clear()
                out = model(x_sw, x_flux, y_p)
                if gate_outputs:
                    g = torch.sigmoid(gate_outputs[0]).mean(dim=-1).numpy()
                    sf = batch.get("storm_flag", torch.zeros(x_sw.shape[0])).numpy().ravel()
                    sf = sf[:g.shape[0]].astype(bool)
                    gate_storm_vals.extend(g[sf].tolist())
                    gate_quiet_vals.extend(g[~sf].tolist())

        for h in hooks:
            h.remove()

        if gate_storm_vals and gate_quiet_vals:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(gate_quiet_vals, bins=40, alpha=0.6, color="#2196F3", label=f"Quiet (n={len(gate_quiet_vals)})", density=True)
            ax.hist(gate_storm_vals, bins=40, alpha=0.6, color="#E91E63", label=f"Storm (n={len(gate_storm_vals)})", density=True)
            ax.axvline(np.mean(gate_quiet_vals), color="#2196F3", lw=1.5, ls="--")
            ax.axvline(np.mean(gate_storm_vals),  color="#E91E63", lw=1.5, ls="--")
            ax.set_xlabel("Bz Gate Activation (σ)")
            ax.set_ylabel("Density")
            ax.set_title("Physics Bz Gate Activation: Storm vs Quiet Windows (GOES test)")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig("plots/fig_interp_gate_activation.png", dpi=300)
            plt.close()
            print(f"Gate — storm: {np.mean(gate_storm_vals):.4f} | quiet: {np.mean(gate_quiet_vals):.4f}")
            print("Saved plots/fig_interp_gate_activation.png")
    except Exception as e:
        print(f"  Gate activation skipped: {e}")

# ──────────────────────────────────────────────────────────────────────────
# I. STORM TIME-SERIES PLOT (NB5 content, GOES test set)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("I. STORM TIME-SERIES PLOT")
print("="*72)

WINDOW_HOURS = 240
ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
if ckpt:
    try:
        model = build_storm_bz(n_sw).to(device)
        model = load_ckpt(model, ckpt)

        yt_all, yp_all, yb_all, st_all = predict(model, goes_test_loader)

        thr_95  = np.percentile(yt_all[:, 1], 95)
        s_idx   = np.where(yt_all[:, 1] > thr_95)[0]
        peak    = s_idx[np.argmax(yt_all[s_idx, 1])]
        i0, i1  = max(0, peak - WINDOW_HOURS // 2), min(len(yt_all), peak + WINDOW_HOURS // 2)
        plot_x  = np.arange(i1 - i0)

        plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        COLORS = ["#2196F3", "#E91E63", "#FF9800"]
        for ax, i, h, col in zip(axes, range(3), HORIZON_NAMES, COLORS):
            ax.plot(plot_x, yt_all[i0:i1, i], color="black", lw=1.5, label="True GOES Flux")
            ax.plot(plot_x, yp_all[i0:i1, i], color=col,     lw=1.5, label=f"STORM-PhysNet ({h})", alpha=0.85)
            ax.fill_between(plot_x, yt_all[i0:i1, i], yp_all[i0:i1, i], alpha=0.1, color=col)
            ax.axhline(thr_95, color="gray", ls="--", lw=0.8, alpha=0.6, label="95th pctile")
            ax.set_ylabel("Log Electron Flux"); ax.set_title(f"{h} forecast")
            ax.legend(fontsize=9, frameon=False, loc="upper right"); ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Time Step (hours)")
        fig.suptitle("STORM-PhysNet Forecast During Major Geomagnetic Storm (GOES E>2 MeV)", fontsize=13)
        fig.tight_layout()
        fig.savefig("plots/fig_timeseries_storm.png", bbox_inches="tight")
        plt.close()

        # Clean single-panel 6h version for paper
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(plot_x, yt_all[i0:i1, 1], color="black", lw=1.5, label="True GOES Flux (E>2 MeV)")
        ax2.plot(plot_x, yp_all[i0:i1, 1], color="#E91E63", lw=1.5, label="STORM-PhysNet (6 h)")
        ax2.axhline(thr_95, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax2.fill_between(plot_x, yt_all[i0:i1, 1], yp_all[i0:i1, 1], alpha=0.1, color="#E91E63")
        ax2.set_xlabel("Time Step (hours)"); ax2.set_ylabel("Log Electron Flux (E > 2 MeV)")
        ax2.set_title("STORM-PhysNet 6 h Electron Flux Forecast — Major Geomagnetic Storm")
        ax2.legend(frameon=False); ax2.grid(True, alpha=0.2)
        fig2.tight_layout()
        fig2.savefig("plots/fig_timeseries_storm_6h.png", bbox_inches="tight")
        plt.close()
        print("Saved plots/fig_timeseries_storm.png + fig_timeseries_storm_6h.png")
    except Exception as e:
        print(f"  Storm plot skipped: {e}")

# ──────────────────────────────────────────────────────────────────────────
# A–F. GRASP EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────
# KEY FIX: GRASP has only flux data — NO solar wind features.
# We join it with the same OMNI solar wind we already have so the
# model sees 16 real SW features (not 2 garbage ones).
# This makes zero-shot and transfer learning physically meaningful.
try:
    from src.data.cdf_reader import read_grasp_directory
    grasp_flux_raw = read_grasp_directory("datasets/grasp")
    assert not grasp_flux_raw.empty, "GRASP directory is empty — no .txt files read"
    print(f"GRASP flux: {len(grasp_flux_raw):,} raw rows ({grasp_flux_raw.index[0].date()} – {grasp_flux_raw.index[-1].date()})")

    # Join GRASP flux with OMNI solar wind (already loaded as `wind` above)
    # Resample GRASP 5-min flux to hourly mean to align with OMNI
    grasp_flux_h = grasp_flux_raw.resample("1h").mean()
    grasp_raw = grasp_flux_h.join(wind, how="inner")
    print(f"GRASP+OMNI joined: {len(grasp_raw):,} hourly rows | cols={list(grasp_raw.columns)}")

    grasp_pre = Preprocessor()
    g_train, g_val, g_test = grasp_pre.fit_transform(grasp_raw)
    _, _, grasp_test_loader = make_dataloaders(
        g_train, g_val, g_test,
        seq_len=SEQ_LEN, batch_size=128, storm_weight=10.0, num_workers=0
    )
    grasp_train_loader, grasp_val_loader, _ = make_dataloaders(
        g_train, g_val, g_test,
        seq_len=SEQ_LEN, batch_size=64, storm_weight=10.0, num_workers=0
    )
    n_sw_grasp = grasp_test_loader.dataset.n_sw_features
    print(f"\nGRASP test set: {len(grasp_test_loader.dataset)} windows | n_sw={n_sw_grasp}")
    if n_sw_grasp != n_sw:
        print(f"  NOTE: GRASP n_sw={n_sw_grasp} vs GOES n_sw={n_sw} — input layers will be adapted by load_ckpt")
    GRASP_AVAILABLE = True
except Exception as e:
    print(f"\nGRASP data unavailable ({e}) — skipping GRASP experiments A–F.")
    print("Run this notebook again after uploading grasp.zip to Drive.")
    GRASP_AVAILABLE = False

grasp_summary = {}

if GRASP_AVAILABLE:
    # ── Fine-tuning helper ────────────────────────────────────────────────
    def fine_tune(model, train_loader, val_loader, epochs, lr, freeze_encoder=False,
                  ckpt_path=None, tag=""):
        if freeze_encoder:
            # Freeze everything except final prediction heads AND the new input layers
            for name, p in model.named_parameters():
                if any(k in name for k in ["flux_head", "storm_head", "var_head", "input_proj", "prop_delay.tau_cond_net"]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            print(f"  [{tag}] Frozen encoder — only training prediction heads")
        else:
            for p in model.parameters():
                p.requires_grad = True
            print(f"  [{tag}] Full fine-tune — all parameters unlocked")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{tag}] Trainable params: {trainable:,}")

        opt       = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.MSELoss()
        best_val  = float("inf")
        best_state= None

        for ep in range(1, epochs + 1):
            model.train()
            tr_losses = []
            for batch in train_loader:
                x_sw   = torch.nan_to_num(batch["x_sw"].to(device),   nan=0.0)
                x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
                y_p    = batch["y_persist"].to(device)
                yt     = batch["y_flux"].to(device)
                opt.zero_grad()
                try:    out = model(x_sw, x_flux, y_p)
                except TypeError: out = model(x_sw, x_flux)
                yp = out["flux_pred"] if isinstance(out, dict) else out
                loss = criterion(yp, yt)
                if torch.isnan(loss): continue
                loss.backward(); opt.step()
                tr_losses.append(loss.item())
            # Validate
            model.eval(); val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x_sw   = torch.nan_to_num(batch["x_sw"].to(device),   nan=0.0)
                    x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
                    y_p    = batch["y_persist"].to(device)
                    yt     = batch["y_flux"].to(device)
                    try:    out = model(x_sw, x_flux, y_p)
                    except TypeError: out = model(x_sw, x_flux)
                    yp = out["flux_pred"] if isinstance(out, dict) else out
                    val_losses.append(criterion(yp, yt).item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val:
                best_val   = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if ckpt_path:
                    torch.save(best_state, ckpt_path)
            if ep % 5 == 0:
                print(f"    [{tag}] Ep {ep:3d} | Train: {np.mean(tr_losses):.4f} | Val: {val_loss:.4f}")
        if best_state:
            model.load_state_dict(best_state)
        return model

    # ── A. Zero-shot: storm_bz ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("A. Zero-shot: GOES storm_bz → GRASP (no fine-tune)")
    print("="*72)
    ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
    if ckpt:
        model = build_storm_bz(n_sw_grasp).to(device)
        model = load_ckpt(model, ckpt)
        yt, yp, yb, st = predict(model, grasp_test_loader)
        grasp_summary["zero_shot_bz"] = horizon_metrics(yt, yp, yb, st, "zero_shot_bz")
        m = grasp_summary["zero_shot_bz"]
        print(f"  PE_45m={fmt(m['pe_45min'])} | PE_6h={fmt(m['pe_6h'])} | PE_12h={fmt(m['pe_12h'])} | PE_storm={fmt(m['pe_storm_6h'])} | PE_hi={fmt(m['pe_highflux_6h'])} | storm_windows={m['n_storm_windows']}")

    # ── B. Zero-shot: transformer ──────────────────────────────────────────
    print("\nB. Zero-shot: GOES transformer → GRASP (no fine-tune)")
    ckpt = find_ckpt(f"checkpoints/transformer/seed_42")
    if ckpt:
        model = build_transformer(n_sw_grasp).to(device)
        model = load_ckpt(model, ckpt)
        yt, yp, yb, st = predict(model, grasp_test_loader)
        grasp_summary["zero_shot_tf"] = horizon_metrics(yt, yp, yb, st, "zero_shot_tf")
        m = grasp_summary["zero_shot_tf"]
        print(f"  PE_45m={fmt(m['pe_45min'])} | PE_6h={fmt(m['pe_6h'])} | PE_12h={fmt(m['pe_12h'])} | PE_storm={fmt(m['pe_storm_6h'])} | PE_hi={fmt(m['pe_highflux_6h'])} | storm_windows={m['n_storm_windows']}")

    # ── C. Frozen TL: storm_bz (heads only) ───────────────────────────────
    print("\nC. Frozen TL: storm_bz — freeze encoder, train heads only")
    ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
    if ckpt:
        frozen_ckpt_dir = Path("checkpoints/grasp_frozen_tl")
        frozen_ckpt_dir.mkdir(exist_ok=True)
        frozen_ckpt_path = frozen_ckpt_dir / "storm_bz_frozen_best.pt"
        model = build_storm_bz(n_sw_grasp).to(device)
        model = load_ckpt(model, ckpt)
        model = fine_tune(model, grasp_train_loader, grasp_val_loader,
                          GRASP_EPOCHS_FROZEN, GRASP_LR_FROZEN,
                          freeze_encoder=True, ckpt_path=frozen_ckpt_path, tag="frozen_TL")
        yt, yp, yb, st = predict(model, grasp_test_loader)
        grasp_summary["frozen_tl"] = horizon_metrics(yt, yp, yb, st, "frozen_tl")
        m = grasp_summary["frozen_tl"]
        print(f"  PE_45m={fmt(m['pe_45min'])} | PE_6h={fmt(m['pe_6h'])} | PE_12h={fmt(m['pe_12h'])} | PE_storm={fmt(m['pe_storm_6h'])} | PE_hi={fmt(m['pe_highflux_6h'])} | storm_windows={m['n_storm_windows']}")

    # ── D. Full TL: storm_bz (all layers) ─────────────────────────────────
    print("\nD. Full fine-tune: storm_bz — all weights unlocked")
    ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
    if ckpt:
        full_ckpt_dir  = Path("checkpoints/grasp_full_tl")
        full_ckpt_dir.mkdir(exist_ok=True)
        full_ckpt_path = full_ckpt_dir / "storm_bz_full_best.pt"
        model = build_storm_bz(n_sw_grasp).to(device)
        model = load_ckpt(model, ckpt)
        model = fine_tune(model, grasp_train_loader, grasp_val_loader,
                          GRASP_EPOCHS_FULL, GRASP_LR_FULL,
                          freeze_encoder=False, ckpt_path=full_ckpt_path, tag="full_TL")
        yt, yp, yb, st = predict(model, grasp_test_loader)
        grasp_summary["full_tl"] = horizon_metrics(yt, yp, yb, st, "full_tl")
        m = grasp_summary["full_tl"]
        print(f"  PE_45m={fmt(m['pe_45min'])} | PE_6h={fmt(m['pe_6h'])} | PE_12h={fmt(m['pe_12h'])} | PE_storm={fmt(m['pe_storm_6h'])} | PE_hi={fmt(m['pe_highflux_6h'])} | storm_windows={m['n_storm_windows']}")

    # ── E. Scratch: train storm_bz on GRASP only ──────────────────────────
    print("\nE. Scratch: storm_bz trained on GRASP only (lower bound / baseline)")
    scratch_ckpt_dir  = Path("checkpoints/grasp_scratch")
    scratch_ckpt_dir.mkdir(exist_ok=True)
    scratch_ckpt_path = scratch_ckpt_dir / "storm_bz_scratch_best.pt"
    model = build_storm_bz(n_sw_grasp).to(device)
    model = fine_tune(model, grasp_train_loader, grasp_val_loader,
                      GRASP_EPOCHS_SCRATCH, GRASP_LR_SCRATCH,
                      freeze_encoder=False, ckpt_path=scratch_ckpt_path, tag="scratch")
    yt, yp, yb, st = predict(model, grasp_test_loader)
    grasp_summary["scratch"] = horizon_metrics(yt, yp, yb, st, "scratch")
    m = grasp_summary["scratch"]
    print(f"  PE_45m={fmt(m['pe_45min'])} | PE_6h={fmt(m['pe_6h'])} | PE_12h={fmt(m['pe_12h'])} | PE_storm={fmt(m['pe_storm_6h'])} | PE_hi={fmt(m['pe_highflux_6h'])} | storm_windows={m['n_storm_windows']}")

    # ── F. Few-shot fine-tune (10% / 50% / 100% of GRASP train) ──────────
    print("\nF. Few-shot TL: frozen encoder, varying GRASP data fraction")
    ckpt = find_ckpt(f"checkpoints/storm_bz/seed_42")
    if ckpt:
        full_train_dataset = grasp_train_loader.dataset
        for frac in FEW_SHOT_FRACS:
            n_samples = max(1, int(len(full_train_dataset) * frac))
            idx = np.random.choice(len(full_train_dataset), n_samples, replace=False)
            subset_loader = DataLoader(Subset(full_train_dataset, idx), batch_size=64, shuffle=True)
            model = build_storm_bz(n_sw_grasp).to(device)
            model = load_ckpt(model, ckpt)
            model = fine_tune(model, subset_loader, grasp_val_loader,
                              GRASP_EPOCHS_FROZEN, GRASP_LR_FROZEN,
                              freeze_encoder=True, tag=f"fewshot_{int(frac*100)}pct")
            yt, yp, yb, st = predict(model, grasp_test_loader)
            row = horizon_metrics(yt, yp, yb, st, f"fewshot_{int(frac*100)}pct")
            grasp_summary[f"fewshot_{int(frac*100)}pct"] = row
            print(f"  {int(frac*100):3d}% GRASP data → PE_6h={fmt(row['pe_6h'])} | PE_storm={fmt(row['pe_storm_6h'])} | PE_hi={fmt(row['pe_highflux_6h'])}")

    # ──────────────────────────────────────────────────────────────────────
    # GRASP FIGURES
    # ──────────────────────────────────────────────────────────────────────

    # Fig: Domain gap bar chart (zero-shot vs frozen TL vs full TL vs scratch)
    key_map = {
        "zero_shot_tf": "Zero-shot\n(Transformer)",
        "zero_shot_bz": "Zero-shot\n(storm_bz)",
        "frozen_tl":    "Frozen TL\n(storm_bz)",
        "full_tl":      "Full TL\n(storm_bz)",
        "scratch":      "Scratch\n(GRASP only)",
    }
    present = {k: v for k, v in key_map.items() if k in grasp_summary}
    if present:
        x = np.arange(len(present)); w = 0.28
        pe6_vals  = [grasp_summary[k]["pe_6h"]          for k in present]
        # PE_storm may be NaN if no storm windows exist in GRASP — use 0 for plotting
        pest_vals = [grasp_summary[k]["pe_storm_6h"]    for k in present]
        pest_vals = [v if not np.isnan(v) else 0.0 for v in pest_vals]
        pehi_vals = [grasp_summary[k]["pe_highflux_6h"] for k in present]
        pehi_vals = [v if not np.isnan(v) else 0.0 for v in pehi_vals]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w, pe6_vals,   w, label="PE all (6h)",       color="#2196F3")
        ax.bar(x,     pest_vals,  w, label="PE storm (6h) [0=no storms]", color="#E91E63")
        ax.bar(x + w, pehi_vals,  w, label="PE high-flux (6h)",  color="#FF9800")
        ax.set_xticks(x); ax.set_xticklabels(list(present.values()), fontsize=9)
        ax.set_ylabel("Prediction Efficiency"); ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title("GRASP (Indian Satellite) — Domain Transfer Performance at 6 h Forecast")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig("plots/fig_grasp_domain_gap.png", dpi=300)
        plt.close()
        print("Saved plots/fig_grasp_domain_gap.png")

    # Fig: Per-horizon PE on GRASP (frozen TL)
    if "frozen_tl" in grasp_summary:
        m = grasp_summary["frozen_tl"]
        pe_horizons = [m["pe_45min"], m["pe_6h"], m["pe_12h"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["45 min", "6 h", "12 h"], pe_horizons, color=["#2196F3", "#E91E63", "#FF9800"], width=0.5)
        ax.set_ylabel("Prediction Efficiency"); ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title("STORM-PhysNet → GRASP: Per-Horizon PE (Frozen Encoder Fine-Tune)")
        for i, v in enumerate(pe_horizons):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig("plots/fig_grasp_horizon.png", dpi=300)
        plt.close()
        print("Saved plots/fig_grasp_horizon.png")

    # Fig: Few-shot data-efficiency curve
    fewshot_keys = [f"fewshot_{int(f*100)}pct" for f in FEW_SHOT_FRACS]
    fewshot_rows = [grasp_summary[k] for k in fewshot_keys if k in grasp_summary]
    if len(fewshot_rows) > 1:
        fracs_pct = [int(f * 100) for f in FEW_SHOT_FRACS[:len(fewshot_rows)]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fracs_pct, [r["pe_6h"]       for r in fewshot_rows], "o-", color="#2196F3", label="PE all (6h)")
        ax.plot(fracs_pct, [r["pe_storm_6h"]  for r in fewshot_rows], "s--",color="#E91E63", label="PE storm (6h)")
        ax.set_xlabel("% of GRASP Training Data Used")
        ax.set_ylabel("Prediction Efficiency")
        ax.set_title("Data Efficiency — Frozen Encoder Transfer to GRASP")
        ax.set_xticks(fracs_pct); ax.legend(frameon=False); ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig("plots/fig_grasp_fewshot.png", dpi=300)
        plt.close()
        print("Saved plots/fig_grasp_fewshot.png")

    # ── GRASP Table (LaTeX-ready) ──────────────────────────────────────────
    sep = "-" * 90
    table_lines = [
        "GRASP Transfer Table — STORM-PhysNet → Indian Satellite (GSAT-12R)",
        "PE = Prediction Efficiency  |  TL = Transfer Learning",
        sep,
        f"{'Experiment':<28} {'PE_45m':>7} {'PE_6h':>7} {'PE_12h':>7} {'PE_storm_6h':>12} {'PE_hi_6h':>9}",
        sep,
    ]
    all_exp_keys = list(key_map.keys()) + fewshot_keys
    for k in all_exp_keys:
        if k not in grasp_summary: continue
        row  = grasp_summary[k]
        name = key_map.get(k, k).replace("\n", " ")
        table_lines.append(
            f"{name:<28} {fmt(row['pe_45min']):>7} {fmt(row['pe_6h']):>7} {fmt(row['pe_12h']):>7} "
            f"{fmt(row['pe_storm_6h']):>12} {fmt(row['pe_highflux_6h']):>9}"
        )
    table_lines.append(sep)
    table_text = "\n".join(table_lines)
    Path("logs/nb4/grasp_table.txt").write_text(table_text)
    print("\n" + table_text)

    json.dump(grasp_summary, open("logs/nb4/summary_grasp.json", "w"), indent=2)

# ── Sync everything to Drive ──────────────────────────────────────────────
print(f"\nSyncing outputs to {DRIVE_NB4_OUT} ...")
for src_path, dst_name in [
    (Path("plots"),    "plots"),
    (Path("logs/nb4"), "logs"),
    (Path("checkpoints/grasp_frozen_tl"),  "grasp_frozen_tl"),
    (Path("checkpoints/grasp_full_tl"),    "grasp_full_tl"),
    (Path("checkpoints/grasp_scratch"),    "grasp_scratch"),
]:
    if src_path.exists():
        dst = Path(DRIVE_NB4_OUT) / dst_name
        shutil.copytree(src_path, dst, dirs_exist_ok=True)

print(f"""
NB4+5 COMPLETE — all outputs synced to {DRIVE_NB4_OUT}
─────────────────────────────────────────────────────
Interpretability figures:
  fig_interp_delay_hist.png       — learned propagation delay
  fig_interp_gate_activation.png  — Bz gate: storm vs quiet
Storm time-series:
  fig_timeseries_storm.png        — 3-panel (45m/6h/12h)
  fig_timeseries_storm_6h.png     — clean 6h panel for IEEE paper
GRASP transfer figures:
  fig_grasp_domain_gap.png        — zero-shot vs fine-tuned bar chart
  fig_grasp_horizon.png           — per-horizon PE on GRASP
  fig_grasp_fewshot.png           — data efficiency curve
Tables:
  grasp_table.txt                 — LaTeX-ready GRASP results
  summary_grasp.json              — all GRASP metrics
─────────────────────────────────────────────────────
""")


# ======================================================================
# --- PHASE: 05_colab_storm_plot.py ---
# ======================================================================

# ============================================================
# NB5 — STORM TIME-SERIES PLOT (Google Colab / IEEE Figure)
# Generates the most important qualitative IEEE figure:
# True vs Predicted electron flux during a major storm event.
#
# Requires NB3 to have run (needs checkpoints + preprocessor.pkl on Drive).
# ============================================================
import os, glob, shutil, pickle, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------- USER SETTINGS --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_NB1_OUT  = "/content/drive/MyDrive/storm_physnet/nb1_outputs"
DRIVE_NB5_OUT  = "/content/drive/MyDrive/storm_physnet/nb5_outputs"

# Which model to plot (use the NB1 winner)
PLOT_LABEL = "storm_bz"
SEED       = 42
SEQ_LEN    = 72

# How many hours around the storm peak to show
WINDOW_HOURS = 240  # 10 days
# -------------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime -> Change runtime type -> GPU"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml matplotlib")

# -------------------- Unpack code --------------------
code_zip = Path(DRIVE_CODE_ZIP)
assert code_zip.exists()
with zipfile.ZipFile(code_zip, "r") as z:
    z.extractall(WORK / "_code")
hits = list((WORK / "_code").rglob("run_training.py"))
code_root = hits[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# -------------------- Unpack data --------------------
dst_goes = WORK / "datasets" / "goes"
dst_omni = WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    with zipfile.ZipFile(Path(DRIVE_DATA_ZIP), "r") as z:
        z.extractall(WORK / "_data")
    g = next((p for p in (WORK / "_data").rglob("goes") if p.is_dir()), None)
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    assert g and o
    if dst_goes.exists(): shutil.rmtree(dst_goes)
    if dst_omni.exists(): shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)

# -------------------- Restore checkpoint --------------------
Path("checkpoints").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

for p in Path(DRIVE_NB1_OUT).rglob(f"seed_{SEED}"):
    if p.parent.name == PLOT_LABEL:
        dest = WORK / "checkpoints" / PLOT_LABEL / f"seed_{SEED}"
        if not dest.exists():
            shutil.copytree(p, dest)
            print(f"Restored {PLOT_LABEL}/seed_{SEED}")
        break

ckpt_path = next(Path(f"checkpoints/{PLOT_LABEL}/seed_{SEED}").glob("*_best.pt"), None)
assert ckpt_path is not None, f"No checkpoint for {PLOT_LABEL} seed {SEED}"

# -------------------- Load data + preprocessor --------------------
from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.storm_physnet import STORMPhysNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goes = read_goes_directory("datasets/goes")
wind = read_wind_directory("datasets/omni")
raw = goes.join(wind, how="inner")

pre_path = next(Path("checkpoints").rglob("preprocessor.pkl"), None) or \
           next(Path(DRIVE_NB1_OUT).rglob("preprocessor.pkl"), None)
if pre_path:
    pre = pickle.load(open(pre_path, "rb"))
    print("preprocessor:", pre_path)
    try:
        train_df, val_df, test_df = pre.fit_transform(raw)
    except Exception:
        train_df, val_df, test_df = Preprocessor().fit_transform(raw)
else:
    train_df, val_df, test_df = Preprocessor().fit_transform(raw)

_, _, test_loader = make_dataloaders(
    train_df, val_df, test_df, seq_len=SEQ_LEN,
    batch_size=256, storm_weight=1.0, num_workers=0
)

# -------------------- Build and load model --------------------
n_sw = test_loader.dataset.n_sw_features
model = STORMPhysNet(
    n_sw_features=n_sw, seq_len=SEQ_LEN, d_model=128, n_heads=4,
    n_transformer_layers=2, n_ssm_layers=2, d_state=64, d_ff=256,
    hidden_dim=64, n_horizons=3, dropout=0.1,
    ablation="none", backbone="transformer", gate_type="bz",
).to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
model.eval()
print(f"Loaded {ckpt_path.name}")

# -------------------- Run inference on full test set --------------------
trues_45m, trues_6h, trues_12h = [], [], []
preds_45m, preds_6h, preds_12h = [], [], []
time_index = []

with torch.no_grad():
    for batch in test_loader:
        x_sw   = torch.nan_to_num(batch["x_sw"].to(device), nan=0.0)
        x_flux = torch.nan_to_num(batch["x_flux"].to(device), nan=0.0)
        y_p    = batch["y_persist"].to(device)
        out    = model(x_sw, x_flux, y_p)
        yp     = out["flux_pred"] if isinstance(out, dict) else out  # [B, 3]
        yt     = batch["y_flux"]                                       # [B, 3]

        trues_45m.append(yt[:, 0].numpy()); preds_45m.append(yp[:, 0].cpu().numpy())
        trues_6h.append( yt[:, 1].numpy()); preds_6h.append( yp[:, 1].cpu().numpy())
        trues_12h.append(yt[:, 2].numpy()); preds_12h.append(yp[:, 2].cpu().numpy())

        if hasattr(batch, "keys") and "time" in batch:
            time_index.extend(batch["time"])

trues_6h = np.concatenate(trues_6h)
preds_6h = np.concatenate(preds_6h)
trues_45m = np.concatenate(trues_45m)
preds_45m = np.concatenate(preds_45m)
trues_12h = np.concatenate(trues_12h)
preds_12h = np.concatenate(preds_12h)

# -------------------- Find biggest storm in test set --------------------
# We use the 95th percentile of the 6h true flux as the storm threshold
thr_95 = np.percentile(trues_6h, 95)
storm_indices = np.where(trues_6h > thr_95)[0]
assert len(storm_indices) > 0, "No major storm found in test set!"

peak_idx = storm_indices[np.argmax(trues_6h[storm_indices])]
start_idx = max(0, peak_idx - WINDOW_HOURS // 2)
end_idx   = min(len(trues_6h), peak_idx + WINDOW_HOURS // 2)
n_plot    = end_idx - start_idx

# Build a simple integer time axis (hours) if we don't have real timestamps
if time_index and len(time_index) == len(trues_6h):
    plot_x = pd.to_datetime(time_index)[start_idx:end_idx]
    use_datetime = True
else:
    plot_x = np.arange(n_plot)
    use_datetime = False

# -------------------- Publication-quality figure --------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
})

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

horizon_data = [
    ("45 min forecast", trues_45m, preds_45m, "#2196F3"),
    ("6 h forecast",    trues_6h,  preds_6h,  "#E91E63"),
    ("12 h forecast",   trues_12h, preds_12h, "#FF9800"),
]

for ax, (title, yt, yp, color) in zip(axes, horizon_data):
    ax.plot(plot_x, yt[start_idx:end_idx], color="black", lw=1.5, label="True GOES Flux", zorder=3)
    ax.plot(plot_x, yp[start_idx:end_idx], color=color,   lw=1.5, label=f"STORM-PhysNet ({title})", alpha=0.85)
    ax.axhline(thr_95, color="gray", ls="--", lw=0.8, alpha=0.6, label="95th percentile")
    ax.fill_between(plot_x, yt[start_idx:end_idx], yp[start_idx:end_idx],
                    alpha=0.12, color=color)
    ax.set_ylabel("Log Electron Flux")
    ax.set_title(title)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)

if use_datetime:
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=25)
    axes[-1].set_xlabel("Date (UTC)")
else:
    axes[-1].set_xlabel("Test set time step (hours)")

fig.suptitle(f"STORM-PhysNet Flux Forecast During Major Storm Event\n"
             f"(storm_bz model, GEO E>2 MeV electrons, peak at index {peak_idx})",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("plots/fig5_storm_timeseries.png", bbox_inches="tight")
print("Saved plots/fig5_storm_timeseries.png")

# Also save individual 6h panel (cleaner version for IEEE submission)
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(plot_x, trues_6h[start_idx:end_idx], color="black", lw=1.5, label="True GOES Flux (E>2 MeV)")
ax2.plot(plot_x, preds_6h[start_idx:end_idx], color="#E91E63", lw=1.5, label="STORM-PhysNet (6 h forecast)")
ax2.axhline(thr_95, color="gray", ls="--", lw=0.8, alpha=0.6, label="95th percentile threshold")
ax2.set_ylabel("Log Electron Flux (E > 2 MeV)")
ax2.set_title("STORM-PhysNet 6 h Electron Flux Forecast During Major Geomagnetic Disturbance")
ax2.legend(frameon=False, loc="upper right")
ax2.grid(True, alpha=0.25)
if use_datetime:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=25)
fig2.tight_layout()
fig2.savefig("plots/fig5b_storm_6h_only.png", bbox_inches="tight")
print("Saved plots/fig5b_storm_6h_only.png  ← use this one in the paper")

# Sync to Drive
Path(DRIVE_NB5_OUT).mkdir(parents=True, exist_ok=True)
shutil.copytree("plots", Path(DRIVE_NB5_OUT) / "plots", dirs_exist_ok=True)
print(f"\nNB5 COMPLETE — plots synced to {DRIVE_NB5_OUT}")
print("Use plots/fig5b_storm_6h_only.png in your IEEE paper.")

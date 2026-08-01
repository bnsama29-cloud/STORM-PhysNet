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
# --- PHASE: 10_final_ieee_eval_fixed.py ---
# ======================================================================

# ============================================================
# 10_FINAL_IEEE_EVAL_FIXED — Google Colab (T4 OK)
# ============================================================
# Purpose
#   One focused evaluation pass over EXISTING checkpoints.
#   Fixes the critical bugs found in 09_colab_eval_script.py
#   and produces only the analyses that support the paper claims.
#
# Bugs fixed vs 09_colab_eval_script.py
#   1. Storm mask used Dst <= -50 on STANDARDIZED Dst (~N(0,1)).
#      That never fired → all "storm" rows missing / identical to quiet.
#      FIX: use batch storm_flag if present; else inverse-scale Dst;
#      else Kp threshold; else high-flux quantile fallback.
#   2. GRASP fine-tune was only 3 epochs on a 256-row tail split.
#      Numbers contradicted the paper. FIX: evaluate zero-shot always;
#      fine-tune only if GRASP data is present, with sane epochs + split;
#      if GRASP is still tiny, mark results as exploratory in the export.
#   3. Label discovery produced names like storm_physnet_storm_cathode.
#      FIX: normalize labels for tables.
#   4. discussion_metrics averaged horizons and had empty storm_pe.
#      FIX: explicit 6 h PE_all / PE_storm / PE_quiet / PE_highflux.
#   5. Permutation used feature indices only.
#      FIX: map to dataset feature names when available.
#   6. Compute cost lacked parameter counts / baseline comparison.
#      FIX: params + latency for Transformer and STORM.
#
# Scope (matches the focused co-author list — nothing extra)
#   01 Setup
#   02 Benchmark (multi-seed PE / RMSE)
#   03 Horizon analysis
#   04 Transfer learning (GRASP) — careful / flagged if weak
#   05 Ablation (with storm PE after mask fix)
#   06 Statistical tests (bootstrap CI, paired seed comparison)
#   07 Physics validation (tau, gate)
#   08 Permutation importance (named features)
#   09 Case studies (worst errors + storm windows)
#   10 Residual analysis
#   11 Uncertainty (MC dropout)
#   12 Compute cost
#   13 Discussion metrics
#   14 Export IEEE tables / figures
#
# Does NOT train new models. Does NOT add t-SNE / attention /
# MiniPatch / hyperparam sweeps / extra seeds.
# ============================================================

# -------------------- USER SETTINGS --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_CKPT_ROOTS = [
    "/content/drive/MyDrive/storm_physnet/nb1_outputs/checkpoints",
    "/content/drive/MyDrive/storm_physnet/nb2_outputs/checkpoints",
    "/content/drive/MyDrive/storm_physnet/ablation_outputs/checkpoints",
]
DRIVE_OUT = "/content/drive/MyDrive/storm_physnet/nb_final_ieee_eval"

# GRASP fine-tune (only if grasp/ exists). Keep modest; paper GRASP
# numbers should still come from your original NB4 if they were better.
DO_GRASP_FINETUNE = True
GRASP_EPOCHS = 15
GRASP_LR = 1e-4

# MC dropout passes for uncertainty band
MC_PASSES = 15
N_BOOTSTRAP = 2000
# -------------------------------------------------------

import os, sys, re, time, json, pickle, shutil, zipfile, warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

assert torch.cuda.is_available(), "Enable GPU: Runtime → T4"
DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work_final_eval")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
OUT = Path(DRIVE_OUT)
for sub in ["Figures", "Tables", "JSON", "LaTeX"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib seaborn")

# ============================================================
# 01 SETUP — code + data
# ============================================================
print("\n=== 01 SETUP ===")
code_work = WORK / "code"
if not (code_work / "src").exists():
    assert Path(DRIVE_CODE_ZIP).exists(), DRIVE_CODE_ZIP
    if code_work.exists():
        shutil.rmtree(code_work)
    code_work.mkdir(parents=True)
    with zipfile.ZipFile(DRIVE_CODE_ZIP) as z:
        z.extractall(code_work)

def find_code_root(root: Path) -> Path:
    if (root / "src").exists() and (root / "configs").exists():
        return root
    for child in root.iterdir():
        if child.is_dir() and (child / "src").exists() and (child / "configs").exists():
            return child
    raise FileNotFoundError(f"No src/configs under {root}")

REPO = find_code_root(code_work)
sys.path.insert(0, str(REPO))
for p in [
    "src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
    "src/training/__init__.py", "src/evaluation/__init__.py",
]:
    Path(REPO / p).parent.mkdir(parents=True, exist_ok=True)
    Path(REPO / p).touch(exist_ok=True)

# data
ds = WORK / "datasets"
for key in ["goes", "omni", "grasp"]:
    if not (ds / key).exists():
        if not Path(DRIVE_DATA_ZIP).exists():
            if key == "grasp":
                continue
            raise FileNotFoundError(DRIVE_DATA_ZIP)
        tmp = WORK / "_data_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        with zipfile.ZipFile(DRIVE_DATA_ZIP) as z:
            z.extractall(tmp)
        found = [p for p in tmp.rglob(key) if p.is_dir()]
        if found:
            ds.mkdir(parents=True, exist_ok=True)
            shutil.copytree(found[0], ds / key, dirs_exist_ok=True)
            print("Copied", key, "<-", found[0])
        if tmp.exists():
            shutil.rmtree(tmp)

import yaml
cfg = yaml.safe_load(open(REPO / "configs" / "config.yaml"))
cfg.setdefault("data", {})
cfg["data"]["goes_cdf_dir"] = str(ds / "goes")
cfg["data"]["wind_cdf_dir"] = str(ds / "omni")
if (ds / "grasp").exists():
    cfg["data"]["grasp_cdf_dir"] = str(ds / "grasp")

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import StandardLSTM, StandardMLP, StandardCNN, VanillaTransformer
from src.model.storm_physnet import STORMPhysNet

try:
    from src.data.cdf_reader import read_grasp_directory
except Exception:
    read_grasp_directory = None

goes = read_goes_directory(str(ds / "goes"))
wind = read_wind_directory(str(ds / "omni"))
raw = goes.join(wind, how="inner")
print("[Data] joined", raw.shape)

# Prefer saved preprocessor from NB1
pp_files = []
for root in DRIVE_CKPT_ROOTS:
    pp_files += list(Path(root).rglob("preprocessor.pkl")) if Path(root).exists() else []
if pp_files:
    try:
        pre = pickle.load(open(sorted(pp_files)[0], "rb"))
        print("Loaded preprocessor:", sorted(pp_files)[0])
        train_df, val_df, test_df = pre.fit_transform(raw)
    except Exception as e:
        print(f"Failed to load preprocessor ({e}). Refitting new one...")
        pre = Preprocessor()
        train_df, val_df, test_df = pre.fit_transform(raw)
else:
    pre = Preprocessor()
    train_df, val_df, test_df = pre.fit_transform(raw)
    print("Fit new preprocessor")

print("split", len(train_df), len(val_df), len(test_df))

SEQ = int(cfg["data"].get("sequence_length", 72))
BS = int(cfg["data"].get("batch_size", 64))
SW = float(cfg["training"].get("storm_weight", 10.0))

try:
    _, _, test_loader = make_dataloaders(
        train_df, val_df, test_df, seq_len=SEQ, batch_size=BS,
        storm_weight=SW, num_workers=0,
    )
except TypeError:
    loaders = make_dataloaders(train_df, val_df, test_df, cfg)
    if isinstance(loaders, dict):
        test_loader = loaders.get("test") or loaders["test_loader"]
    else:
        test_loader = loaders[2]

N_SW = getattr(test_loader.dataset, "n_sw_features", None)
SW_COLS = list(getattr(test_loader.dataset, "sw_cols", []) or [])
if N_SW is None:
    b0 = next(iter(test_loader))
    x = b0["x_sw"] if isinstance(b0, dict) else b0[0]
    N_SW = int(x.shape[-1])
print(f"n_sw={N_SW}  sw_cols={SW_COLS[:8]}{'...' if len(SW_COLS) > 8 else ''}  test_windows={len(test_loader.dataset)}")

HORIZONS = ["45min", "6h", "12h"]
H6 = 1  # 6 h index

# ============================================================
# Helpers: model build, checkpoint discovery, metrics, storm mask
# ============================================================

def normalize_label(raw_label: str) -> str:
    """Map messy discovery names → clean paper labels."""
    s = raw_label.lower()
    # order matters
    if "no_delay" in s:
        return "storm_no_delay"
    if "no_physics" in s:
        return "storm_no_physics"
    if "cathode" in s and "spec" in s:
        return "storm_cathode_spec"
    if "cathode" in s:
        return "storm_cathode"
    if "radio" in s and "spec" in s:
        return "storm_radio_spec"
    if "radio" in s:
        return "storm_radiotrophic"
    if "bz_mag" in s or s.endswith("storm_bz") or "/storm_bz" in s or s == "storm_bz":
        return "storm_bz"
    if s in ("storm_physnet", "storm_physnet_best") or s.endswith("storm_physnet"):
        return "storm_bz"  # main STORM model in this project
    if s == "transformer":
        return "transformer"
    if s in ("lstm", "mlp", "cnn"):
        return s
    return raw_label


def build_model(label: str):
    label = normalize_label(label)
    if label == "lstm":
        return StandardLSTM(n_sw_features=N_SW, seq_len=SEQ, n_horizons=3)
    if label == "mlp":
        return StandardMLP(n_sw_features=N_SW, seq_len=SEQ, n_horizons=3)
    if label == "cnn":
        return StandardCNN(n_sw_features=N_SW, seq_len=SEQ, n_horizons=3)
    if label == "transformer":
        return VanillaTransformer(n_sw_features=N_SW, seq_len=SEQ, n_horizons=3)

    gate = "bz"
    if "cathode" in label:
        gate = "cathode_anode"
    elif "radio" in label:
        gate = "radiotrophic"
    ablation = "none"
    if "no_delay" in label:
        ablation = "no_delay"
    if "no_physics" in label:
        ablation = "no_physics"
    use_spec = "spec" in label

    return STORMPhysNet(
        n_sw_features=N_SW,
        seq_len=SEQ,
        d_model=int(cfg["model"].get("d_model", 128)),
        n_heads=int(cfg["model"].get("transformer", {}).get("n_heads", 4)),
        n_transformer_layers=int(cfg["model"].get("transformer", {}).get("n_layers", 3)),
        n_ssm_layers=int(cfg["model"].get("ssm", {}).get("n_layers", 2)),
        d_state=int(cfg["model"].get("ssm", {}).get("d_state", 64)),
        d_ff=int(cfg["model"].get("transformer", {}).get("d_ff", 256)),
        hidden_dim=int(cfg["model"].get("heads", {}).get("hidden_dim", 64)),
        n_horizons=3,
        dropout=float(cfg["model"].get("transformer", {}).get("dropout", 0.1)),
        ablation=ablation,
        backbone="transformer",
        gate_type=gate,
        use_spectral_head=use_spec,
    )


def load_checkpoint(model, path):
    state = torch.load(str(path), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(str(k).startswith("member_0") for k in state):
        state = {k.replace("member_0.", ""): v for k, v in state.items() if str(k).startswith("member_0")}
    model.load_state_dict(state, strict=False)
    return model


def discover_checkpoints():
    """Return {clean_label: {seed: path}}."""
    mapping = defaultdict(dict)
    roots = [Path(r) for r in DRIVE_CKPT_ROOTS if Path(r).exists()]
    if not roots and (Path.cwd() / "checkpoints").exists():
        roots = [Path.cwd() / "checkpoints"]
    for root in roots:
        for fp in list(root.rglob("*_best.pt")) + list(root.rglob("*_best.zip")):
            if not fp.is_file():
                continue
            path_str = str(fp)
            if any(x in path_str for x in ["sweep_", "seqlen_", "nb3_outputs", "nb4_outputs", "STALE"]):
                continue
            # seed from path
            m = re.search(r"seed[_]?(\d+)", path_str)
            seed = int(m.group(1)) if m else 0
            # label from parent folders preferentially
            parts = fp.parts
            label_raw = fp.stem.replace("_best", "")
            for i, p in enumerate(parts):
                if p.startswith("seed_") and i > 0:
                    label_raw = parts[i - 1]
                    break
            label = normalize_label(label_raw)
            # prefer first-seen; allow overwrite if same seed later from ablation root
            mapping[label][seed] = fp
    return {k: dict(v) for k, v in mapping.items()}


def pe_vs_persist(y_true, y_pred, y_persist):
    """PE = 1 - MSE_model / MSE_persistence (per horizon if 2D)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_persist = np.asarray(y_persist)
    if y_true.ndim == 1:
        mse_m = np.mean((y_true - y_pred) ** 2)
        mse_p = np.mean((y_true - y_persist) ** 2)
        return float(1.0 - mse_m / (mse_p + 1e-12))
    mse_m = np.mean((y_true - y_pred) ** 2, axis=0)
    mse_p = np.mean((y_true - y_persist) ** 2, axis=0)
    return 1.0 - mse_m / (mse_p + 1e-12)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def inverse_dst_if_scaled(dst_scaled, preprocessor):
    """Try to recover physical Dst [nT] from standardized values."""
    dst_scaled = np.asarray(dst_scaled, dtype=float)
    # Common patterns on this project
    scaler = getattr(preprocessor, "scaler", None)
    cols = getattr(preprocessor, "feature_cols", None) or getattr(preprocessor, "all_features", None)
    if scaler is not None and cols is not None:
        cols = list(cols)
        for name in ["dst", "Dst", "DST", "sym_h", "SYM_H"]:
            if name in cols:
                i = cols.index(name)
                mean = float(getattr(scaler, "mean_", [0] * len(cols))[i])
                scale = float(getattr(scaler, "scale_", [1] * len(cols))[i])
                return dst_scaled * scale + mean
    # Heuristic: if |dst| median < 5, values are standardized → approximate
    # typical Dst std ~ 30–40 nT; mean near 0. Use 35 nT if unknown.
    if np.nanmedian(np.abs(dst_scaled)) < 5:
        return dst_scaled * 35.0
    return dst_scaled


def build_masks(y_true_6h, dst, kp, storm_flag, preprocessor):
    """
    Return dict of boolean masks on the test set.
    Priority for storm:
      1) storm_flag from dataloader
      2) inverse-scaled Dst <= -50 nT
      3) Kp >= 5 (if Kp looks physical)
      4) high-flux top 10% (PE_hi proxy — not geomagnetic storm)
    """
    n = len(y_true_6h)
    masks = {"all": np.ones(n, dtype=bool)}

    if storm_flag is not None and len(storm_flag) == n and storm_flag.any():
        masks["storm"] = storm_flag.astype(bool)
        masks["quiet"] = ~masks["storm"]
        masks["storm_source"] = "storm_flag"
    else:
        dst_phys = inverse_dst_if_scaled(dst, preprocessor) if dst is not None else None
        if dst_phys is not None and np.nanmin(dst_phys) < -20:
            masks["storm"] = dst_phys <= -50.0
            masks["quiet"] = ~masks["storm"]
            masks["storm_source"] = "dst_physical_le_m50"
        elif kp is not None and np.nanmax(kp) > 3.5:
            # Kp often left unscaled or lightly scaled
            masks["storm"] = kp >= 5.0
            masks["quiet"] = ~masks["storm"]
            masks["storm_source"] = "kp_ge_5"
        else:
            # Fallback: high-flux regime (NOT the same as geomagnetic storm)
            thr = np.nanpercentile(y_true_6h, 90)
            masks["storm"] = y_true_6h >= thr
            masks["quiet"] = ~masks["storm"]
            masks["storm_source"] = "high_flux_p90_fallback"
            print("WARNING: using high-flux p90 as storm proxy — report as PE_hi, not PE_storm")

    # Always also report high-flux (paper PE_hi)
    thr_hi = np.nanpercentile(y_true_6h, 90)
    masks["high_flux"] = y_true_6h >= thr_hi
    return masks


@torch.no_grad()
def predict(model, loader, mc_dropout=False, mc_passes=1, record_aux=False):
    model.to(DEVICE)
    model.train(mc_dropout)
    preds_pass, trues, persists, dsts, kps, storms, gates, taus = [], [], [], [], [], [], [], []

    for p in range(mc_passes if mc_dropout else 1):
        batch_preds = []
        first = (p == 0)
        for batch in loader:
            if isinstance(batch, dict):
                x_sw = torch.nan_to_num(batch["x_sw"].to(DEVICE), nan=0.0)
                x_flux = torch.nan_to_num(batch.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1)).to(DEVICE), nan=0.0)
                y_persist = batch["y_persist"].to(DEVICE)
                try:
                    out = model(x_sw, x_flux, y_persist=y_persist)
                except TypeError:
                    try:
                        out = model(x_sw, x_flux, y_persist)
                    except TypeError:
                        out = model(x_sw, x_flux)
                pred = out["flux_pred"] if isinstance(out, dict) else out
                batch_preds.append(pred.cpu().numpy())
                if first:
                    trues.append(batch["y_flux"].numpy())
                    persists.append(batch["y_persist"].numpy())
                    if "y_dst" in batch:
                        d = batch["y_dst"].numpy()
                        dsts.append(d.min(axis=1) if d.ndim == 2 else d.ravel())
                    if "y_kp" in batch:
                        k = batch["y_kp"].numpy()
                        kps.append(k.max(axis=1) if k.ndim == 2 else k.ravel())
                    if "storm_flag" in batch:
                        storms.append(batch["storm_flag"].numpy().ravel())
                    if record_aux and isinstance(out, dict):
                        if "gate_values" in out:
                            gates.append(out["gate_values"].cpu().numpy())
                        if "tau" in out:
                            taus.append(out["tau"].cpu().numpy())
            else:
                # tuple fallback
                x, y = batch[0].to(DEVICE).float(), batch[1].to(DEVICE).float()
                out = model(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                batch_preds.append(out.cpu().numpy())
                if first:
                    trues.append(y.cpu().numpy())
                    persists.append(y.cpu().numpy())  # weak fallback
        preds_pass.append(np.concatenate(batch_preds, 0))

    yt = np.concatenate(trues, 0)
    yb = np.concatenate(persists, 0)
    stack = np.stack(preds_pass, 0)  # [P, N, H]
    yp = stack.mean(0)
    ystd = stack.std(0) if mc_dropout else None
    dst = np.concatenate(dsts, 0) if dsts else None
    kp = np.concatenate(kps, 0) if kps else None
    st = np.concatenate(storms, 0).astype(bool) if storms else None
    gate = np.concatenate(gates, 0) if gates else None
    tau = np.concatenate(taus, 0) if taus else None
    model.eval()
    return yt, yp, yb, dst, kp, st, gate, tau, ystd


def metrics_block(yt, yp, yb, masks, label, seed):
    rows = []
    for period, m in masks.items():
        if period == "storm_source":
            continue
        if m is None or m.sum() < 5:
            continue
        for h, hname in enumerate(HORIZONS):
            pe = pe_vs_persist(yt[m, h], yp[m, h], yb[m, h])
            rows.append({
                "label": label,
                "seed": seed,
                "horizon": hname,
                "period": period,
                "pe": float(pe),
                "rmse": rmse(yt[m, h], yp[m, h]),
                "mae": float(np.mean(np.abs(yp[m, h] - yt[m, h]))),
                "r2": float(r2_score(yt[m, h], yp[m, h])) if m.sum() > 1 else 0.0,
                "n": int(m.sum()),
                "storm_source": masks.get("storm_source", "unknown"),
            })
    return rows


def savefig(name):
    path = OUT / "Figures" / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", path.name)
    return path


# ============================================================
# Discover checkpoints + cache predictions
# ============================================================
print("\n=== Discover checkpoints ===")
CKPT = discover_checkpoints()
print("Labels:", sorted(CKPT.keys()))
for lab, seeds in CKPT.items():
    print(f"  {lab}: seeds={sorted(seeds.keys())}")

assert CKPT, "No checkpoints found — check DRIVE_CKPT_ROOTS"

# Prefer evaluating these labels for the paper
PRIMARY = [l for l in [
    "transformer", "storm_bz", "storm_no_delay", "storm_no_physics",
    "storm_cathode", "storm_cathode_spec", "storm_radiotrophic",
    "lstm", "mlp", "cnn",
] if l in CKPT]

print("PRIMARY labels for tables:", PRIMARY)

pred_cache = {}  # (label, seed) -> dict
print("\n=== Running inference (cached) ===")
for label in PRIMARY:
    for seed, path in sorted(CKPT[label].items()):
        try:
            model = build_model(label)
            load_checkpoint(model, path)
            yt, yp, yb, dst, kp, st, gate, tau, _ = predict(model, test_loader, record_aux=("storm" in label or label == "storm_bz"))
            masks = build_masks(yt[:, H6], dst, kp, st, pre)
            if seed == sorted(CKPT[label].keys())[0]:
                print(f"  {label}: storm_source={masks.get('storm_source')}  "
                      f"storm_frac={masks['storm'].mean():.3f}  n_storm={masks['storm'].sum()}")
            pred_cache[(label, seed)] = {
                "yt": yt, "yp": yp, "yb": yb, "dst": dst, "kp": kp, "st": st,
                "gate": gate, "tau": tau, "masks": masks, "path": str(path),
            }
            print(f"  cached {label} seed={seed}")
        except Exception as e:
            print(f"  FAIL {label} seed={seed}: {e}")

assert pred_cache, "No successful predictions"

# ============================================================
# 02 BENCHMARK
# ============================================================
print("\n=== 02 BENCHMARK ===")
bench_rows = []
for (label, seed), d in pred_cache.items():
    bench_rows += metrics_block(d["yt"], d["yp"], d["yb"], d["masks"], label, seed)
bench = pd.DataFrame(bench_rows)
bench.to_csv(OUT / "Tables" / "benchmark_metrics.csv", index=False)
print(bench.groupby(["label", "horizon", "period"])["pe"].mean().unstack(fill_value=np.nan).round(4))

# ============================================================
# 03 HORIZON ANALYSIS
# ============================================================
print("\n=== 03 HORIZON ANALYSIS ===")
fig, ax = plt.subplots(figsize=(7, 4.2))
for label in [l for l in ["transformer", "storm_bz"] if l in CKPT]:
    means, stds = [], []
    for h in HORIZONS:
        sub = bench[(bench.label == label) & (bench.horizon == h) & (bench.period == "all")]
        means.append(sub.pe.mean())
        stds.append(sub.pe.std(ddof=1) if len(sub) > 1 else 0.0)
    ax.errorbar(HORIZONS, means, yerr=stds, marker="o", capsize=4, label=label)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("PE (mean ± std over seeds)")
ax.set_title("Multi-horizon prediction efficiency")
ax.legend()
fig.tight_layout()
savefig("fig_horizon_pe.png")

# ============================================================
# 05 ABLATION (with storm PE)
# ============================================================
print("\n=== 05 ABLATION ===")
ablation_labels = [l for l in ["storm_bz", "storm_no_delay", "storm_no_physics"] if l in CKPT]
abl_rows = []
for label in ablation_labels:
    sub = bench[bench.label == label]
    abl_rows.append(sub)
if abl_rows:
    abl = pd.concat(abl_rows, ignore_index=True)
    abl.to_csv(OUT / "Tables" / "ablation_metrics.csv", index=False)
    # bar chart PE_all and PE_storm at 6h
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, period, title in zip(axes, ["all", "storm"], ["PE_all (6 h)", "PE_storm (6 h)"]):
        vals, errs, names = [], [], []
        for label in ablation_labels:
            s = abl[(abl.label == label) & (abl.horizon == "6h") & (abl.period == period)]
            if s.empty:
                continue
            names.append(label.replace("storm_", ""))
            vals.append(s.pe.mean())
            errs.append(s.pe.std(ddof=1) if len(s) > 1 else 0.0)
        ax.bar(names, vals, yerr=errs, capsize=4, color="steelblue")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    savefig("fig_ablation_6h.png")

# ============================================================
# 06 STATISTICAL TESTS
# ============================================================
print("\n=== 06 STATISTICAL TESTS ===")
rng = np.random.default_rng(0)
stat_report = {}
for label in [l for l in ["transformer", "storm_bz"] if l in CKPT]:
    pes = []
    for seed in sorted(CKPT[label].keys()):
        if (label, seed) not in pred_cache:
            continue
        d = pred_cache[(label, seed)]
        pes.append(float(pe_vs_persist(d["yt"][:, H6], d["yp"][:, H6], d["yb"][:, H6])))
    pes = np.asarray(pes, float)
    boots = [rng.choice(pes, size=len(pes), replace=True).mean() for _ in range(N_BOOTSTRAP)] if len(pes) else []
    stat_report[label] = {
        "n_seeds": int(len(pes)),
        "pe_6h_values": pes.tolist(),
        "mean": float(pes.mean()) if len(pes) else None,
        "sample_std": float(pes.std(ddof=1)) if len(pes) > 1 else 0.0,
        "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
    }
    print(label, stat_report[label])

if "transformer" in stat_report and "storm_bz" in stat_report:
    a = np.asarray(stat_report["storm_bz"]["pe_6h_values"])
    b = np.asarray(stat_report["transformer"]["pe_6h_values"])
    n = min(len(a), len(b))
    if n >= 2:
        d = a[:n] - b[:n]
        null = []
        for _ in range(5000):
            signs = rng.choice([-1.0, 1.0], size=n)
            null.append((signs * d).mean())
        p = float(np.mean(np.abs(null) >= abs(d.mean())))
        # Cohen's d (paired)
        cohens = float(d.mean() / (d.std(ddof=1) + 1e-12))
        stat_report["paired_storm_minus_tf"] = {
            "mean_diff": float(d.mean()),
            "sign_flip_p": p,
            "cohens_d": cohens,
            "note": "Exploratory: n_seeds is small; report cautiously.",
        }
        print("ΔPE", d.mean(), "sign-flip p", p, "Cohen d", cohens)
json.dump(stat_report, open(OUT / "JSON" / "statistics.json", "w"), indent=2)

# ============================================================
# 07 PHYSICS VALIDATION
# ============================================================
print("\n=== 07 PHYSICS VALIDATION ===")
delay_rows = []
for label in [l for l in ["storm_bz", "storm_no_delay", "storm_no_physics"] if l in CKPT]:
    for seed, d in [(s, pred_cache[(label, s)]) for s in CKPT[label] if (label, s) in pred_cache]:
        tau = d.get("tau")
        if tau is None:
            # try reading from model parameters
            model = build_model(label)
            load_checkpoint(model, d["path"])
            for n, p in model.named_parameters():
                if "tau" in n.lower() or "delay" in n.lower():
                    delay_rows.append({"label": label, "seed": seed, "tau_h": float(p.detach().cpu().reshape(-1)[0]), "source": n})
                    break
        else:
            delay_rows.append({"label": label, "seed": seed, "tau_h": float(np.mean(tau)), "source": "forward"})
if delay_rows:
    pd.DataFrame(delay_rows).to_csv(OUT / "Tables" / "physics_tau.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label in sorted(set(r["label"] for r in delay_rows)):
        vals = [r["tau_h"] for r in delay_rows if r["label"] == label]
        ax.hist(vals, bins=max(3, len(vals)), alpha=0.6, label=label)
    ax.set_xlabel("Learned τ (hours)")
    ax.set_title("Propagation delay across seeds / variants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig("fig_physics_tau_hist.png")

# Gate activation storm vs quiet for main model
if ("storm_bz", min(CKPT.get("storm_bz", {42: None}))) in pred_cache or "storm_bz" in CKPT:
    seed0 = sorted(CKPT["storm_bz"].keys())[0] if "storm_bz" in CKPT else None
    if seed0 is not None and ( "storm_bz", seed0) in pred_cache:
        d = pred_cache[("storm_bz", seed0)]
        if d["gate"] is not None:
            g = d["gate"]
            if g.ndim > 1:
                g = g.mean(axis=tuple(range(1, g.ndim)))
            m_st = d["masks"]["storm"][: len(g)]
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(g[~m_st], bins=40, alpha=0.6, label="quiet", density=True)
            if m_st.any():
                ax.hist(g[m_st], bins=40, alpha=0.6, label="storm", density=True)
            ax.set_xlabel("Gate activation")
            ax.set_title("Bz-gate activation: storm vs quiet")
            ax.legend()
            fig.tight_layout()
            savefig("fig_physics_gate_storm_quiet.png")

# ============================================================
# 08 PERMUTATION IMPORTANCE
# ============================================================
print("\n=== 08 PERMUTATION IMPORTANCE ===")
focus = "storm_bz" if "storm_bz" in CKPT else PRIMARY[0]
seed0 = sorted(CKPT[focus].keys())[0]
model = build_model(focus)
load_checkpoint(model, CKPT[focus][seed0])
model.to(DEVICE).eval()

# baseline
d0 = pred_cache[(focus, seed0)]
base_pe = float(pe_vs_persist(d0["yt"][:, H6], d0["yp"][:, H6], d0["yb"][:, H6]))
print("baseline PE_6h", base_pe)

feat_names = SW_COLS if len(SW_COLS) == N_SW else [f"f{i}" for i in range(N_SW)]
imp_rows = []
for fi, fname in enumerate(feat_names):
    # permute feature fi across batch dimension for each batch
    ys, ps, bs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_sw = batch["x_sw"].clone()
            perm = torch.randperm(x_sw.shape[0])
            x_sw[:, :, fi] = x_sw[perm, :, fi]
            x_sw = torch.nan_to_num(x_sw.to(DEVICE), nan=0.0)
            x_flux = torch.nan_to_num(batch["x_flux"].to(DEVICE), nan=0.0)
            y_persist = batch["y_persist"].to(DEVICE)
            try:
                out = model(x_sw, x_flux, y_persist=y_persist)
            except TypeError:
                try:
                    out = model(x_sw, x_flux, y_persist)
                except TypeError:
                    out = model(x_sw, x_flux)
            pred = out["flux_pred"] if isinstance(out, dict) else out
            ps.append(pred.cpu().numpy())
            ys.append(batch["y_flux"].numpy())
            bs.append(batch["y_persist"].numpy())
    yt = np.concatenate(ys); yp = np.concatenate(ps); yb = np.concatenate(bs)
    pe_p = float(pe_vs_persist(yt[:, H6], yp[:, H6], yb[:, H6]))
    imp_rows.append({"feature": fname, "index": fi, "pe_permuted": pe_p, "pe_drop": base_pe - pe_p})
    if fi % 4 == 0:
        print(f"  {fname}: drop={base_pe - pe_p:.4f}")

imp = pd.DataFrame(imp_rows).sort_values("pe_drop", ascending=False)
imp.to_csv(OUT / "Tables" / "permutation_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(imp["feature"][::-1], imp["pe_drop"][::-1], color="#2E5C8A")
ax.set_xlabel("PE drop when permuted (6 h)")
ax.set_title(f"Permutation importance — {focus}")
fig.tight_layout()
savefig("fig_feature_importance.png")

# ============================================================
# 09 CASE STUDIES
# ============================================================
print("\n=== 09 CASE STUDIES ===")
d = pred_cache[(focus, seed0)]
yt, yp, yb = d["yt"][:, H6], d["yp"][:, H6], d["yb"][:, H6]
err = np.abs(yp - yt)
# worst 5 absolute errors
top = np.argsort(err)[-5:][::-1]
# one high-flux / storm window center
st_m = d["masks"]["storm"]
storm_centers = np.where(st_m)[0]
center_storm = int(storm_centers[len(storm_centers) // 2]) if len(storm_centers) else int(np.argmax(yt))
center_worst = int(top[0])

def panel(ax, center, title, half=72):
    a, b = max(0, center - half), min(len(yt), center + half)
    t = np.arange(a, b)
    ax.plot(t, yt[a:b], label="true", lw=1.4)
    ax.plot(t, yp[a:b], label="pred", lw=1.2)
    ax.plot(t, yb[a:b], label="persist", lw=1.0, alpha=0.7)
    ax.set_title(title)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
panel(axes[0], center_storm, "Case — storm / high-flux neighborhood (6 h)")
panel(axes[1], center_worst, "Case — largest absolute error neighborhood (6 h)")
axes[1].set_xlabel("Test index")
fig.tight_layout()
savefig("fig_case_studies.png")

pd.DataFrame([{
    "index": int(i),
    "true_6h": float(yt[i]),
    "pred_6h": float(yp[i]),
    "persist_6h": float(yb[i]),
    "abs_err": float(err[i]),
} for i in top]).to_csv(OUT / "Tables" / "event_case_studies.csv", index=False)

# ============================================================
# 10 RESIDUAL ANALYSIS
# ============================================================
print("\n=== 10 RESIDUAL ANALYSIS ===")
for label in [l for l in ["transformer", "storm_bz"] if l in CKPT]:
    seed = sorted(CKPT[label].keys())[0]
    d = pred_cache[(label, seed)]
    resid = d["yp"][:, H6] - d["yt"][:, H6]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(d["yt"][:, H6], resid, s=3, alpha=0.25)
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_xlabel("True log-flux"); axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{label} residual scatter (6 h)")
    axes[1].hist(resid, bins=60, alpha=0.85)
    axes[1].set_title(f"{label} residual histogram (6 h)")
    fig.tight_layout()
    savefig(f"fig_residual_{label}.png")

# ============================================================
# 11 UNCERTAINTY (MC dropout)
# ============================================================
print("\n=== 11 UNCERTAINTY ===")
model = build_model(focus)
load_checkpoint(model, CKPT[focus][seed0])
yt, yp, yb, dst, kp, st, _, _, ystd = predict(model, test_loader, mc_dropout=True, mc_passes=MC_PASSES)
# coverage under Gaussian approx: |err| <= 1.645 std
cover = float(np.mean(np.abs(yt[:, H6] - yp[:, H6]) <= 1.645 * (ystd[:, H6] + 1e-8)))
print(f"MC dropout coverage@90%≈{cover:.3f}  mean_std={ystd[:, H6].mean():.4f}")
json.dump({
    "model": focus, "seed": seed0, "mc_passes": MC_PASSES,
    "coverage_90_approx": cover, "mean_pred_std_6h": float(ystd[:, H6].mean()),
}, open(OUT / "JSON" / "mc_dropout.json", "w"), indent=2)

n = min(300, len(yt))
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(yt[:n, H6], color="k", lw=1, label="true")
ax.plot(yp[:n, H6], color="C3", lw=1, label="MC mean")
ax.fill_between(np.arange(n), yp[:n, H6] - 1.645 * ystd[:n, H6], yp[:n, H6] + 1.645 * ystd[:n, H6],
                 color="C3", alpha=0.25, label="~90% band")
ax.legend(fontsize=8); ax.set_title(f"MC-dropout band — {focus} 6 h")
fig.tight_layout()
savefig("fig_mc_dropout_band.png")

# ============================================================
# 12 COMPUTE COST
# ============================================================
print("\n=== 12 COMPUTE COST ===")
cost_rows = []
for label in [l for l in ["transformer", "storm_bz"] if l in CKPT]:
    model = build_model(label).to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    # synthetic batch
    x_sw = torch.randn(32, SEQ, N_SW, device=DEVICE)
    x_flux = torch.randn(32, SEQ, 1, device=DEVICE)
    y_p = torch.randn(32, 3, device=DEVICE)
    with torch.no_grad():
        for _ in range(10):
            try:
                model(x_sw, x_flux, y_persist=y_p)
            except TypeError:
                try:
                    model(x_sw, x_flux, y_p)
                except TypeError:
                    model(x_sw, x_flux)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            try:
                model(x_sw, x_flux, y_persist=y_p)
            except TypeError:
                try:
                    model(x_sw, x_flux, y_p)
                except TypeError:
                    model(x_sw, x_flux)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 50 * 1000
    cost_rows.append({"label": label, "parameters": int(n_params), "ms_per_batch32": round(ms, 3)})
    print(cost_rows[-1])
pd.DataFrame(cost_rows).to_csv(OUT / "Tables" / "compute_cost.csv", index=False)

# ============================================================
# 04 TRANSFER LEARNING (GRASP) — careful
# ============================================================
print("\n=== 04 GRASP TRANSFER ===")
grasp_summary = {"status": "skipped"}
if (ds / "grasp").exists() and read_grasp_directory is not None and "storm_bz" in CKPT:
    try:
        grasp_df = read_grasp_directory(str(ds / "grasp"))
        wind2 = read_wind_directory(str(ds / "omni"))
        grasp_raw = grasp_df.join(wind2, how="inner")
        print("GRASP joined", grasp_raw.shape)
        if len(grasp_raw) < 100:
            grasp_summary = {"status": "too_few_rows", "n": len(grasp_raw)}
            print("GRASP too small — skip fine-tune; do not put these PE numbers in the main paper table")
        else:
            # transform with GOES-fitted preprocessor
            try:
                gdf = pre.transform(grasp_raw)
            except Exception:
                gdf = Preprocessor().fit_transform(grasp_raw)
                if isinstance(gdf, tuple):
                    gdf = pd.concat(list(gdf), axis=0)
            if isinstance(gdf, tuple):
                # some preprocessors return splits only on fit
                gdf = grasp_raw
            n = len(gdf)
            # chronological 70/15/15
            i1, i2 = int(0.70 * n), int(0.85 * n)
            tr, va, te = gdf.iloc[:i1], gdf.iloc[i1:i2], gdf.iloc[i2:]
            try:
                _, _, grasp_loader = make_dataloaders(tr, va, te, seq_len=SEQ, batch_size=BS, storm_weight=1.0, num_workers=0)
            except TypeError:
                grasp_loader = test_loader  # last resort
                print("WARNING: could not build GRASP loader with same API")

            seed0 = sorted(CKPT["storm_bz"].keys())[0]
            model = build_model("storm_bz")
            load_checkpoint(model, CKPT["storm_bz"][seed0])
            # zero-shot
            yt, yp, yb, dst, kp, st, _, _, _ = predict(model, grasp_loader)
            masks_g = build_masks(yt[:, H6], dst, kp, st, pre)
            zs_rows = metrics_block(yt, yp, yb, masks_g, "storm_bz_zeroshot", seed0)
            for r in zs_rows:
                r["stage"] = "zero_shot"
            print("Zero-shot 6h PE_all", [r for r in zs_rows if r["horizon"] == "6h" and r["period"] == "all"])

            ft_rows = []
            if DO_GRASP_FINETUNE and len(te) >= 50:
                # light fine-tune heads (or full — keep simple full with low LR)
                model.train()
                opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=GRASP_LR)
                for ep in range(GRASP_EPOCHS):
                    losses = []
                    for batch in grasp_loader:
                        x_sw = torch.nan_to_num(batch["x_sw"].to(DEVICE), nan=0.0)
                        x_flux = torch.nan_to_num(batch["x_flux"].to(DEVICE), nan=0.0)
                        y = batch["y_flux"].to(DEVICE)
                        y_persist = batch["y_persist"].to(DEVICE)
                        try:
                            out = model(x_sw, x_flux, y_persist=y_persist)
                        except TypeError:
                            try:
                                out = model(x_sw, x_flux, y_persist)
                            except TypeError:
                                out = model(x_sw, x_flux)
                        pred = out["flux_pred"] if isinstance(out, dict) else out
                        loss = nn.functional.mse_loss(pred, y)
                        opt.zero_grad(); loss.backward(); opt.step()
                        losses.append(loss.item())
                    if (ep + 1) % 5 == 0:
                        print(f"  GRASP ft epoch {ep+1}/{GRASP_EPOCHS} loss={np.mean(losses):.4f}")
                model.eval()
                yt2, yp2, yb2, dst2, kp2, st2, _, _, _ = predict(model, grasp_loader)
                masks2 = build_masks(yt2[:, H6], dst2, kp2, st2, pre)
                ft_rows = metrics_block(yt2, yp2, yb2, masks2, "storm_bz_finetuned", seed0)
                for r in ft_rows:
                    r["stage"] = "finetuned"
                torch.save(model.state_dict(), OUT / "Tables" / "storm_bz_grasp_finetuned.pt")

            gdf_out = pd.DataFrame(zs_rows + ft_rows)
            gdf_out.to_csv(OUT / "Tables" / "grasp_transfer_comparison.csv", index=False)
            grasp_summary = {
                "status": "ok",
                "n_test_rows_approx": int(len(te)),
                "zero_shot_6h_pe_all": float(gdf_out[(gdf_out.stage == "zero_shot") & (gdf_out.horizon == "6h") & (gdf_out.period == "all")]["pe"].mean()) if len(gdf_out) else None,
                "finetuned_6h_pe_all": float(gdf_out[(gdf_out.stage == "finetuned") & (gdf_out.horizon == "6h") & (gdf_out.period == "all")]["pe"].mean()) if any(gdf_out.stage == "finetuned") else None,
                "warning": "If PE is far below paper NB4 numbers, keep paper GRASP table from NB4 and treat this as a reproducibility check only.",
            }
            # simple bar if both stages exist
            if any(gdf_out.stage == "finetuned"):
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                stages = ["zero_shot", "finetuned"]
                vals = []
                for stg in stages:
                    s = gdf_out[(gdf_out.stage == stg) & (gdf_out.horizon == "6h") & (gdf_out.period == "all")]
                    vals.append(s.pe.mean() if len(s) else np.nan)
                ax.bar(stages, vals, color=["#999", "#2E5C8A"])
                ax.set_ylabel("PE_all (6 h)")
                ax.set_title("GRASP transfer (this notebook)")
                fig.tight_layout()
                savefig("fig_grasp_transfer.png")
    except Exception as e:
        grasp_summary = {"status": "error", "error": str(e)}
        print("GRASP failed:", e)
else:
    print("GRASP data or reader missing — skip")
json.dump(grasp_summary, open(OUT / "JSON" / "grasp_summary.json", "w"), indent=2)

# ============================================================
# 13 DISCUSSION METRICS
# ============================================================
print("\n=== 13 DISCUSSION METRICS ===")
disc = []
for label in [l for l in ["transformer", "storm_bz", "storm_no_delay", "storm_no_physics"] if l in CKPT]:
    sub = bench[(bench.label == label) & (bench.horizon == "6h")]
    def mean_pe(period):
        s = sub[sub.period == period]
        return float(s.pe.mean()) if len(s) else float("nan")
    disc.append({
        "label": label,
        "pe_all_6h": mean_pe("all"),
        "pe_storm_6h": mean_pe("storm"),
        "pe_quiet_6h": mean_pe("quiet"),
        "pe_highflux_6h": mean_pe("high_flux"),
        "n_seeds": int(sub[sub.period == "all"]["seed"].nunique()),
    })
disc_df = pd.DataFrame(disc)
disc_df.to_csv(OUT / "Tables" / "discussion_metrics.csv", index=False)
print(disc_df.to_string(index=False))

# Improvement helper vs transformer
if "transformer" in disc_df.label.values and "storm_bz" in disc_df.label.values:
    tf = disc_df[disc_df.label == "transformer"].iloc[0]
    st = disc_df[disc_df.label == "storm_bz"].iloc[0]
    improvements = {
        "delta_pe_all_6h": float(st.pe_all_6h - tf.pe_all_6h),
        "delta_pe_storm_6h": float(st.pe_storm_6h - tf.pe_storm_6h),
        "delta_pe_quiet_6h": float(st.pe_quiet_6h - tf.pe_quiet_6h),
        "delta_pe_highflux_6h": float(st.pe_highflux_6h - tf.pe_highflux_6h),
    }
    json.dump(improvements, open(OUT / "JSON" / "discussion_improvements.json", "w"), indent=2)
    print("Improvements vs Transformer:", improvements)

# ============================================================
# 14 EXPORT IEEE-READY SUMMARY TABLE + LATEX SNIPPET
# ============================================================
print("\n=== 14 EXPORT ===")
# Main seed-mean table at 6h
lines = []
lines.append("Model,n_seeds,PE_all,PE_all_std,PE_storm,PE_storm_std,PE_highflux,RMSE_all")
for label in PRIMARY:
    sub = bench[(bench.label == label) & (bench.horizon == "6h")]
    if sub.empty:
        continue
    def stats(period):
        s = sub[sub.period == period]
        if s.empty:
            return float("nan"), float("nan")
        return float(s.pe.mean()), float(s.pe.std(ddof=1)) if len(s) > 1 else 0.0
    pa, pas = stats("all")
    ps, pss = stats("storm")
    ph, _ = stats("high_flux")
    rm = float(sub[sub.period == "all"].rmse.mean())
    n = int(sub[sub.period == "all"].seed.nunique())
    lines.append(f"{label},{n},{pa:.4f},{pas:.4f},{ps:.4f},{pss:.4f},{ph:.4f},{rm:.4f}")

(OUT / "Tables" / "ieee_main_table.csv").write_text("\n".join(lines))
print("\n".join(lines))

# LaTeX snippet
tex = []
tex.append("% Auto-generated — paste into results section")
tex.append("\\begin{tabular}{lcccc}")
tex.append("\\hline")
tex.append("Model & PE$_{\\mathrm{all}}$ & PE$_{\\mathrm{storm}}$ & PE$_{\\mathrm{hi}}$ & RMSE \\\\")
tex.append("\\hline")
for line in lines[1:]:
    parts = line.split(",")
    tex.append(f"{parts[0]} & {parts[2]}$\\pm${parts[3]} & {parts[4]}$\\pm${parts[5]} & {parts[6]} & {parts[7]} \\\\")
tex.append("\\hline")
tex.append("\\end{tabular}")
(OUT / "LaTeX" / "main_results_snippet.tex").write_text("\n".join(tex))

# Checklist
checklist = {
    "outputs": str(OUT),
    "storm_mask_source": next(iter(pred_cache.values()))["masks"].get("storm_source"),
    "figures": sorted(p.name for p in (OUT / "Figures").glob("*.png")),
    "tables": sorted(p.name for p in (OUT / "Tables").glob("*.csv")),
    "json": sorted(p.name for p in (OUT / "JSON").glob("*.json")),
    "grasp": grasp_summary,
    "paper_use": {
        "safe_for_main_text": [
            "benchmark_metrics.csv (GOES)",
            "fig_horizon_pe.png",
            "fig_residual_*.png",
            "fig_feature_importance.png",
            "fig_physics_tau_hist.png",
            "compute_cost.csv",
            "statistics.json (report p as exploratory if n_seeds small)",
        ],
        "use_only_after_checking_storm_source": [
            "ablation_metrics.csv",
            "discussion_metrics.csv",
            "PE_storm columns",
        ],
        "do_not_overwrite_paper_grasp_unless_better": [
            "grasp_transfer_comparison.csv",
        ],
    },
}
json.dump(checklist, open(OUT / "JSON" / "CHECKLIST.json", "w"), indent=2)
print("\n=== DONE ===")
print(json.dumps(checklist, indent=2))
print(f"\nAll outputs → {OUT}")
print("""
Next steps for the paper:
1) Open JSON/CHECKLIST.json and confirm storm_mask_source is NOT high_flux_p90_fallback
   (if it is, PE_storm is really PE_hi — rename in the paper).
2) Copy Figures/ and Tables/ieee_main_table.csv into Overleaf.
3) Keep previous NB4 GRASP numbers if this GRASP block is weaker.
4) In text, keep claims cautious: 'suggests', 'consistent with', exploratory p-values.
""")

# ============================================================
# FINAL MANUSCRIPT TABLES GENERATOR (TABLE II)
# Run this AFTER all models finish training across all seeds!
# ============================================================
import os
os.system("pip -q install cdflib")
import yaml
import torch
import pickle
import zipfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define Paths ---
DRIVE_ROOT   = Path("/content/drive/MyDrive/storm_physnet")
DRIVE_CODE   = DRIVE_ROOT / "ieee_final_fixed.zip"   
DRIVE_DATA   = DRIVE_ROOT / "datasets.zip"
DRIVE_NB1    = DRIVE_ROOT / "nb1_stats_outputs" / "checkpoints"
DRIVE_OUT    = DRIVE_ROOT / "revision_experiments"

WORK = Path("/content/storm_work")
WORK.mkdir(exist_ok=True)
os.chdir(WORK)

# --- Setup Code & Data if not already present ---
if not (WORK / "src").exists():
    print("Extracting code (fixing Windows paths)...")
    with zipfile.ZipFile(DRIVE_CODE, "r") as z:
        for member in z.namelist():
            # Fix backslashes in zip paths created on Windows
            target_path = WORK / "_code" / member.replace("\\", "/")
            if member.endswith("/") or member.endswith("\\"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(z.read(member))
                
    src_hits = sorted(list((WORK / "_code").rglob("src")), key=lambda p: len(p.parts))
    if not src_hits:
        raise RuntimeError("Still could not find src after extraction fix. Something is very wrong!")
    
    code_root = src_hits[0].parent
    for name in ["src", "configs"]:
        if (code_root / name).is_dir(): shutil.copytree(code_root / name, WORK / name, dirs_exist_ok=True)
        
for key in ["goes", "omni"]:
    if not (WORK / "datasets" / key).exists():
        print(f"Extracting {key} data...")
        with zipfile.ZipFile(DRIVE_DATA, "r") as z: z.extractall(WORK / "_data")
        hits = list((WORK / "_data").rglob(key))
        if hits: shutil.copytree(hits[0], WORK / "datasets" / key, dirs_exist_ok=True)

import sys
sys.path.insert(0, str(WORK))

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import StandardLSTM, VanillaTransformer
from src.model.storm_physnet import STORMPhysNet
from src.evaluation.metrics import prediction_efficiency

def get_model(label, n_sw):
    if label == "lstm":
        return StandardLSTM(n_sw_features=n_sw, seq_len=72, n_horizons=3)
    elif label == "transformer":
        return VanillaTransformer(n_sw_features=n_sw, seq_len=72, n_horizons=3)
    elif label == "no_gate":
        return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_gate")
    elif label == "no_delay":
        return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_delay")
    elif label == "no_physics":
        return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_physics")
    else: # storm_physnet
        return STORMPhysNet(n_sw_features=n_sw, seq_len=72)

def predict(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x_sw = batch[0] if isinstance(batch, (list, tuple)) else batch["x_sw"]
            x_flux = batch[1] if isinstance(batch, (list, tuple)) else batch["x_flux"]
            y = batch[2] if isinstance(batch, (list, tuple)) else batch.get("y_flux", batch.get("y"))
            pers = batch[3] if isinstance(batch, (list, tuple)) else batch.get("y_persist", batch.get("pers"))
            
            x_sw, x_flux = x_sw.to(device), x_flux.to(device)
            out = model(x_sw, x_flux, y_persist=pers.to(device) if pers is not None else None)
            pred = out.get("flux_pred", list(out.values())[0]) if isinstance(out, dict) else out
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

print("\nLoading datasets into memory (this takes ~30 seconds)...")
raw = read_goes_directory("datasets/goes").join(read_wind_directory("datasets/omni"), how="inner")
pp_path = DRIVE_NB1 / "transformer" / "seed_42" / "preprocessor.pkl"
with open(pp_path, "rb") as f: pre = pickle.load(f)

# Hotfix for backward compatibility with old preprocessor files
if not hasattr(pre, "year_split"): pre.year_split = None
if not hasattr(pre, "train_frac"): pre.train_frac = 0.7; pre.val_frac = 0.15

if hasattr(pre, "_split"):
    train_df, val_df, test_df = pre._split(pre.transform(raw))
else:
    train_df, val_df, test_df = pre.fit_transform(raw)

cfg = yaml.safe_load(open("configs/config.yaml"))
_, _, test_loader = make_dataloaders(
    train_df, val_df, test_df, 
    seq_len=cfg["data"].get("sequence_length", 72), 
    batch_size=cfg["training"].get("batch_size", 64)
)
n_sw = next(iter(test_loader))[0].shape[-1] if isinstance(next(iter(test_loader)), (list, tuple)) else next(iter(test_loader))["x_sw"].shape[-1]

# Map models to where they are stored on Drive
models_to_evaluate = {
    "Transformer": ("transformer", DRIVE_NB1),
    "STORM-PhysNet": ("storm_physnet", DRIVE_NB1),
    "STORM (No Delay)": ("storm_no_delay", DRIVE_NB1),
    "STORM (No Physics)": ("storm_no_physics", DRIVE_NB1),
    "STORM (No Gate)": ("no_gate", DRIVE_OUT / "checkpoints"),
    "LSTM": ("lstm", DRIVE_OUT / "checkpoints"),
}

results, all_preds, y_ref = {}, {}, None

for display_name, (label, source_dir) in models_to_evaluate.items():
    print(f"\nEvaluating {display_name}...")
    preds = []
    
    for seed in range(42, 57):
        ckpt_dir = source_dir / label / f"seed_{seed}"
        
        # In NB1, you sometimes saved it as "storm_bz" instead of "storm_physnet"
        if label == "storm_physnet" and not list(ckpt_dir.glob("*_best.pt")):
            ckpt_dir = source_dir / "storm_bz" / f"seed_{seed}"
            
        cands = list(ckpt_dir.glob("*_best.pt"))
        
        # Fallback to tier1_extra_seeds if not found in NB1
        if not cands and source_dir == DRIVE_NB1:
            fallback_dir = DRIVE_ROOT / "tier1_extra_seeds" / "checkpoints"
            ckpt_dir = fallback_dir / label / f"seed_{seed}"
            if label == "storm_physnet" and not list(ckpt_dir.glob("*_best.pt")):
                ckpt_dir = fallback_dir / "storm_bz" / f"seed_{seed}"
            cands = list(ckpt_dir.glob("*_best.pt"))
            
        if not cands:
            print(f"  Seed {seed}: Checkpoint missing! Skipping.")
            continue
            
        print(f"  Seed {seed}: Checkpoint Loaded -> Running Predictions...")
        model = get_model(label, n_sw).to(device)
        state = torch.load(cands[0], map_location=device)
        state = state.get("state_dict", state)
        
        if any(str(k).startswith("member_") for k in state.keys()):
            k0 = [k for k in state if str(k).startswith("member_")][0]
            state = state[k0] if not isinstance(state[k0], dict) else state[k0]
            
        model.load_state_dict(state, strict=False)
        y_t, p_t = predict(model, test_loader)
        preds.append(p_t)
        y_ref = y_t
        
    if not preds: continue
        
    seed_pes_45, seed_pes_6, seed_pes_12 = [], [], []
    for p_t in preds:
        seed_pes_45.append(prediction_efficiency(y_ref[:, 0], p_t[:, 0]))
        seed_pes_6.append(prediction_efficiency(y_ref[:, 1], p_t[:, 1]))
        seed_pes_12.append(prediction_efficiency(y_ref[:, 2], p_t[:, 2]))
        
    ensembled_pred = np.mean(preds, axis=0)
    all_preds[label] = ensembled_pred
    
    results[f"{display_name} (Mean of Seeds)"] = {
        "45-min PE": round(np.mean(seed_pes_45), 3),
        "6-hour PE": round(np.mean(seed_pes_6), 3),
        "12-hour PE": round(np.mean(seed_pes_12), 3)
    }
    
    results[f"{display_name} (Bagged 15-seed)"] = {
        "45-min PE": round(prediction_efficiency(y_ref[:, 0], ensembled_pred[:, 0]), 3),
        "6-hour PE": round(prediction_efficiency(y_ref[:, 1], ensembled_pred[:, 1]), 3),
        "12-hour PE": round(prediction_efficiency(y_ref[:, 2], ensembled_pred[:, 2]), 3)
    }

if "transformer" in all_preds and "storm_physnet" in all_preds:
    print("\nEvaluating Ensembles...")
    
    # Alpha = 0.3 Blend
    blend = 0.3 * all_preds["storm_physnet"] + 0.7 * all_preds["transformer"]
    results["Ensemble (α=0.3 blend)"] = {
        "45-min PE": round(prediction_efficiency(y_ref[:, 0], blend[:, 0]), 3),
        "6-hour PE": round(prediction_efficiency(y_ref[:, 1], blend[:, 1]), 3),
        "12-hour PE": round(prediction_efficiency(y_ref[:, 2], blend[:, 2]), 3)
    }
    
    # Hybrid (Short STORM / Long TF)
    hybrid = all_preds["transformer"].copy()
    hybrid[:, 0] = all_preds["storm_physnet"][:, 0]  # STORM for 45-min
    results["Hybrid (short STORM / long TF)"] = {
        "45-min PE": round(prediction_efficiency(y_ref[:, 0], hybrid[:, 0]), 3),
        "6-hour PE": round(prediction_efficiency(y_ref[:, 1], hybrid[:, 1]), 3),
        "12-hour PE": round(prediction_efficiency(y_ref[:, 2], hybrid[:, 2]), 3)
    }

df = pd.DataFrame.from_dict(results, orient='index')
print("\n\n=== FINAL MANUSCRIPT RESULTS (TABLE II) ===")
print(df.to_markdown())

csv_out = DRIVE_OUT / "tables" / "FINAL_TABLE_II.csv"
csv_out.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(csv_out)
print(f"\nSaved directly to your Google Drive at: {csv_out}")

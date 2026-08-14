# ============================================================
# STORM-PhysNet — Parallel Experiments (FIXED v2)
# ============================================================

# ============================================================
# 🔧 EDIT THESE TWO LINES PER ACCOUNT
# ============================================================
ACCOUNT_ID = 1  # CHANGE THIS: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9

# TASKS assigned to this account
TASKS = [
    ("wider_delay", {"seed": 42, "upper_bound": 1.5, "epochs": 6}),
    ("wider_delay", {"seed": 42, "upper_bound": 2.0, "epochs": 6}),
    ("bagged_tf", {"seed": 42, "epochs": 8}),
]
# ============================================================

# ============================================================
# FORCE numpy<2 BEFORE ANY IMPORTS
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"])
import numpy as np
print("numpy version:", np.__version__)

# ============================================================
# NOW IMPORTS (after numpy pinned)
# ============================================================
import os, shutil, subprocess, yaml, json, sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE = Path("/content/drive/MyDrive/storm_physnet")
CODE_ZIP = DRIVE / "ieee_final_fixed.zip"
DATA_ZIP = DRIVE / "datasets.zip"
OUT = Path("/content/drive/MyDrive/storm_physnet") / f"experiments_account_{ACCOUNT_ID}"
OUT.mkdir(parents=True, exist_ok=True)

os.chdir("/content")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "cdflib", "pandas", "scikit-learn", "pyyaml", "tqdm", "scipy", "matplotlib", "torch"], check=True)

WORK = Path("/content/storm_work")
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir(exist_ok=True)
os.chdir(WORK)

print("Extracting code...")
shutil.copy2(CODE_ZIP, WORK / "code.zip")
subprocess.run(["unzip", "-q", "-o", "code.zip"], cwd=WORK, capture_output=True)
sys.path.insert(0, str(WORK))

for key in ["goes", "omni", "grasp"]:
    dst = Path("datasets") / key
    if not dst.exists():
        alt = Path("/content/drive/MyDrive/storm_physnet/datasets") / key
        if alt.exists(): shutil.copytree(alt, dst)
        else:
            shutil.unpack_archive(str(DATA_ZIP), WORK / "_data")
            hits = list(Path("_data").glob(f"**/{key}"))
            if hits: shutil.copytree(hits[0], dst)

import src
from src.training.trainer import Trainer
from src.model.storm_physnet import STORMPhysNet
from src.model.baselines import VanillaTransformer
from src.data.dataloader import make_dataloaders
from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor

# Load base config
with open("configs/config.yaml") as f:
    base_cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== ACCOUNT {ACCOUNT_ID} | Device: {device} | numpy {np.__version__} ===")
print(f"Tasks assigned: {len(TASKS)}")

# ============================================================
# EXPERIMENT IMPLEMENTATIONS
# ============================================================

def build_dataloaders(cfg):
    """Build train/val/test loaders from config"""
    cfg_data = cfg["data"]
    goes = read_goes_directory(cfg_data["goes_cdf_dir"])
    wind = read_wind_directory(cfg_data["wind_cdf_dir"])
    raw = goes.join(wind, how="inner")
    
    prep_path = None
    for root in [Path("/content/drive/MyDrive/storm_physnet/nb1_stats_outputs/checkpoints"), 
                 Path("/content/drive/MyDrive/storm_physnet/nb1_outputs/checkpoints")]:
        hits = list(root.rglob("preprocessor.pkl"))
        if hits:
            prep_path = hits[0]
            break
    
    if prep_path:
        pre = Preprocessor.load(str(prep_path)) if hasattr(Preprocessor, "load") else pickle.load(open(prep_path, "rb"))
        df = pre.transform(raw)
    else:
        pre = Preprocessor()
        df = pre.fit_transform(raw)
    
    n = len(df)
    a, b = int(0.70 * n), int(0.85 * n)
    train_df, val_df, test_df = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        seq_len=72, batch_size=cfg["data"]["batch_size"],
        storm_weight=cfg["training"]["storm_weight"], num_workers=0
    )
    return train_loader, val_loader, test_loader

def run_wider_delay(seed, upper_bound, epochs):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Wider Delay [0.5, {upper_bound}]h | seed={seed}")
    print(f"{'='*60}")
    
    cfg = base_cfg.copy()
    cfg["model"]["delay_min"] = 0.5
    cfg["model"]["delay_max"] = upper_bound
    cfg["training"]["epochs"] = epochs
    cfg["training"]["seed"] = seed
    
    run_name = f"wider_delay_{upper_bound}h_seed{seed}"
    ckpt_dir = OUT / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = str(OUT / "logs" / run_name)
    
    with open(OUT / f"config_{run_name}.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    
    print(f"Running {run_name}...")
    trainer = Trainer(cfg)
    
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    
    start = time.time()
    try:
        # FIX: use n_sw_features not n_sw
        model = trainer.fit(train_loader, val_loader, n_sw_features=16, use_ensemble=False)
        elapsed = time.time() - start
        results = evaluate_model(model, test_loader, device)
        
        result_file = OUT / f"results_{run_name}.json"
        with open(result_file, "w") as f:
            json.dump({
                "run_name": run_name, "seed": seed, "upper_bound": upper_bound,
                "epochs": epochs, "time_sec": elapsed, "results": results
            }, f, indent=2)
        
        print(f"Completed {run_name} in {elapsed/60:.1f} min")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"ERROR in {run_name}: {e}")
        with open(OUT / f"error_{run_name}.txt", "w") as f:
            f.write(str(e))

def run_bagged_tf(seed, epochs):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: Bagged Transformer | seed={seed}")
    print(f"{'='*60}")
    
    cfg = base_cfg.copy()
    cfg["model_type"] = "transformer"
    cfg["training"]["epochs"] = epochs
    cfg["training"]["seed"] = seed
    
    run_name = f"bagged_tf_seed{seed}"
    ckpt_dir = OUT / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = str(OUT / "logs" / run_name)
    
    with open(OUT / f"config_{run_name}.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    
    # Use base config for data loading
    cfg_data = base_cfg["data"]
    goes = read_goes_directory(cfg_data["goes_cdf_dir"])
    wind = read_wind_directory(cfg_data["wind_cdf_dir"])
    raw = goes.join(wind, how="inner")
    
    prep_path = None
    for root in [Path("/content/drive/MyDrive/storm_physnet/nb1_stats_outputs/checkpoints"), 
                 Path("/content/drive/MyDrive/storm_physnet/nb1_outputs/checkpoints")]:
        hits = list(root.rglob("preprocessor.pkl"))
        if hits:
            prep_path = hits[0]
            break
    
    if prep_path:
        pre = Preprocessor.load(str(prep_path)) if hasattr(Preprocessor, "load") else pickle.load(open(prep_path, "rb"))
        df = pre.transform(raw)
    else:
        pre = Preprocessor()
        df = pre.fit_transform(raw)
    
    n = len(df)
    a, b = int(0.70 * n), int(0.85 * n)
    train_df, val_df, test_df = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        seq_len=72, batch_size=base_cfg["data"]["batch_size"],
        storm_weight=base_cfg["training"]["storm_weight"], num_workers=0
    )
    
    print(f"Running {run_name}...")
    trainer = Trainer(cfg)
    model = trainer.fit(train_loader, val_loader, n_sw_features=16, use_ensemble=False)
    results = evaluate_model(model, test_loader, device)
    
    result_file = OUT / f"results_{run_name}.json"
    with open(result_file, "w") as f:
        json.dump({
            "run_name": run_name, "seed": seed, "epochs": epochs,
            "results": results
        }, f, indent=2)
    print(f"Completed {run_name}")

def evaluate_model(model, test_loader, device):
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_sw = batch["x_sw"].to(device)
            x_flux = batch["x_flux"].to(device)
            y = batch["y_flux"].to(device)
            
            try:
                out = model(x_sw, x_flux, batch["y_persist"].to(device))
            except TypeError:
                out = model(x_sw, x_flux)
            
            if isinstance(out, dict):
                pred = out.get("flux_pred", out.get("pred", None))
            else:
                pred = out
            
            ys.append(y.cpu().numpy())
            yh.append(pred.cpu().numpy())
    
    ys = np.concatenate(ys, 0)
    yh = np.concatenate(yh, 0)
    
    results = {}
    for h, hname in enumerate(["45min", "6h", "12h"]):
        yt = ys[:, h]
        yp = yh[:, h]
        mse = np.mean((yt - yp) ** 2)
        var = np.var(yt)
        pe = 1 - mse / (var + 1e-12)
        results[f"PE_{hname}"] = float(pe)
        results[f"RMSE_{hname}"] = float(np.sqrt(mse))
    
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"Starting Account {ACCOUNT_ID} with {len(TASKS)} tasks")
    
    for task_name, params in TASKS:
        try:
            if task_name == "physics_inspect":
                run_physics_inspect()
            elif task_name == "wider_delay":
                run_wider_delay(**params)
            elif task_name == "bagged_tf":
                run_bagged_tf(**params)
            else:
                print(f"Unknown task: {task_name}")
        except Exception as e:
            print(f"TASK FAILED: {task_name} - {e}")
            with open(OUT / f"error_{task_name}.txt", "w") as f:
                f.write(str(e))
    
    print(f"\n✅ Account {ACCOUNT_ID} COMPLETE")
    print(f"Results saved to: {OUT}")
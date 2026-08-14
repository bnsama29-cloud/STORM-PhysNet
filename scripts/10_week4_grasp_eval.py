# ============================================================
# WEEK 4 — GRASP Transfer Evaluation & Plotting
# Goal: Aggregate results across all seeds and plot the Domain Gap
# ============================================================
import os, sys, shutil, yaml, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

WORK = Path("/content/storm_eval_grasp")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

try: import cdflib
except ImportError: os.system("pip -q install cdflib pandas scikit-learn pyyaml tqdm matplotlib")

# Set up code and data
DRIVE = Path("/content/drive/MyDrive/storm_physnet")
DRIVE_CODE_ZIP = DRIVE / "ieee_final_fixed.zip"
DRIVE_DATA_ZIP = DRIVE / "datasets.zip"
DRIVE_OUT = DRIVE / "week4_grasp_outputs"

_code = WORK / "_code"
if not _code.exists():
    _code.mkdir()
    os.system(f'unzip -q -o "{DRIVE_CODE_ZIP}" -d "{_code}"')

code_root = list(_code.rglob("run_training.py"))[0].parent
for name in ["src", "configs"]:
    s, d = code_root / name, WORK / name
    if s.exists():
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(s, d)

for ds in ["grasp", "omni"]:
    dst = WORK / "datasets" / ds
    if not dst.exists():
        _data = WORK / "_data"
        _data.mkdir(exist_ok=True)
        os.system(f'unzip -q -n "{DRIVE_DATA_ZIP}" -d "{_data}"')
        hits = [p for p in _data.rglob(ds) if p.is_dir()]
        if hits:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(hits[0], dst)

if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
for key in list(sys.modules.keys()):
    if key.startswith("src"): del sys.modules[key]

from src.data.cdf_reader import read_grasp_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.model.storm_physnet import STORMPhysNet

def build_storm(n_sw):
    return STORMPhysNet(n_sw_features=n_sw, seq_len=72, n_horizons=3, gate_type="bz", backbone="transformer")

def load_state(model, path):
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]; break
    model.load_state_dict(ck, strict=False)
    model.eval()
    return model

# Load data
print("\n--- Loading GRASP and OMNI Data ---")
raw_grasp = read_grasp_directory("datasets/grasp")
raw_wind = read_wind_directory("datasets/omni")
raw = raw_grasp.join(raw_wind, how="inner").dropna()

H_IDX = {"1h": 0, "6h": 1, "12h": 2}
def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    return float(1.0 - np.sum((y - yhat)**2) / (np.sum((y - np.mean(y))**2) + 1e-8))

def predict_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x_sw, x_flux = batch["x_sw"].to(device), batch["x_flux"].to(device)
            try: out = model(x_sw, x_flux, batch["y_persist"].to(device))
            except TypeError: out = model(x_sw, x_flux)
                
            if isinstance(out, dict): pred = out.get("flux_pred", out.get("pred", None))
            else: pred = out
            preds.append(pred.cpu().numpy())
            trues.append(batch["y_flux"].numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)

SEEDS = list(range(42, 57))
results = []

for seed in SEEDS:
    base_pts = list(DRIVE.rglob(f"nb1_stats_outputs/checkpoints/storm_bz/seed_{seed}/*_best.pt")) + \
               list(DRIVE.rglob(f"tier1_extra_seeds/checkpoints/storm_bz/seed_{seed}/*_best.pt"))
    if not base_pts: continue
        
    ft_pts = list(DRIVE_OUT.glob(f"seed_{seed}/*_best.pt"))
    if not ft_pts: continue
    
    pre_path = base_pts[0].parent / "preprocessor.pkl"
    pre = Preprocessor.load(str(pre_path)) if hasattr(Preprocessor, "load") else pickle.load(open(pre_path, "rb"))
    if not hasattr(pre, "year_split"): pre.year_split = None
    if not hasattr(pre, "train_frac"): pre.train_frac = 0.70
    if not hasattr(pre, "val_frac"): pre.val_frac = 0.15
    
    train_df, val_df, test_df = pre._split(raw)
    test_df = pre.transform(test_df)
    
    from torch.utils.data import DataLoader
    from src.data.dataloader import FluxDataset
    test_ds = FluxDataset(test_df, seq_len=72)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    batch = next(iter(test_loader))
    n_sw = batch["x_sw"].size(-1) if isinstance(batch, dict) else batch[0].size(-1)

    print(f"[{seed}] Evaluating Zero-Shot vs Fine-Tuned...")
    
    m_base = load_state(build_storm(n_sw).to(device), base_pts[0])
    m_ft   = load_state(build_storm(n_sw).to(device), ft_pts[0])

    y_base, y = predict_loader(m_base, test_loader)
    y_ft, _   = predict_loader(m_ft, test_loader)
    
    for name, j in H_IDX.items():
        results.append({"seed": seed, "model": "Zero-Shot (GOES)", "horizon": name, "PE": pe_clim(y[:, j], y_base[:, j])})
        results.append({"seed": seed, "model": "Fine-Tuned (GRASP)", "horizon": name, "PE": pe_clim(y[:, j], y_ft[:, j])})

if results:
    df = pd.DataFrame(results)
    df.to_csv(DRIVE_OUT / "grasp_metrics_summary.csv", index=False)
    
    g = df.groupby(["model", "horizon"]).mean(numeric_only=True).round(3).reset_index()
    print("\n\n======== GRASP DOMAIN TRANSFER SUMMARY ========")
    print(g.pivot(index="model", columns="horizon", values="PE")[["1h", "6h", "12h"]].to_markdown())
    
    # ---- PLOTTING ----
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        horizons = ["1h", "6h", "12h"]
        zero = [g[(g["model"]=="Zero-Shot (GOES)") & (g["horizon"]==h)]["PE"].values[0] for h in horizons]
        ft = [g[(g["model"]=="Fine-Tuned (GRASP)") & (g["horizon"]==h)]["PE"].values[0] for h in horizons]
        
        x = np.arange(len(horizons))
        width = 0.35
        
        ax.bar(x - width/2, zero, width, label='Zero-Shot (GOES Model)', color='#d62728', edgecolor='black')
        ax.bar(x + width/2, ft, width, label='Fine-Tuned (GRASP Model)', color='#2ca02c', edgecolor='black')
        
        ax.set_ylabel('Prediction Efficiency (PE)', fontsize=12)
        ax.set_title('Cross-Satellite Transfer Learning (GOES → GRASP)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons, fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.0])
        
        for i, (z, f) in enumerate(zip(zero, ft)):
            ax.text(i - width/2, z + 0.02, f"{z:.2f}", ha='center', fontweight='bold', fontsize=10)
            ax.text(i + width/2, f + 0.02, f"{f:.2f}", ha='center', fontweight='bold', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(DRIVE_OUT / "domain_gap_plot.png", dpi=300)
        print("\nPlot successfully saved to:", DRIVE_OUT / "domain_gap_plot.png")
    except Exception as e:
        print("Plotting failed:", e)
else:
    print("No finished GRASP seeds found yet!")

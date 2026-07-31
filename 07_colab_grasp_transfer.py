# ============================================================
# NB7 — GRASP TRANSFER LEARNING (Google Colab / GPU)
# Dedicated script for fine-tuning pre-trained GOES models
# onto the ISRO GSAT-19 (GRASP) dataset.
# Reproduces Table II (GRASP Domain Transfer) from the paper.
# ============================================================

import os, glob, shutil, zipfile, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error

# -------------------- USER SETTINGS --------------------
# Base folder where your datasets.zip, grasp.zip, ieee_final_fixed.zip, and nb1_outputs live.
DRIVE_ROOT = "/content/drive/MyDrive/storm_physnet"
# (If you added the shared folder as a shortcut in your Drive with a different name, change DRIVE_ROOT above)

DRIVE_CODE_ZIP = f"{DRIVE_ROOT}/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = f"{DRIVE_ROOT}/datasets.zip"
DRIVE_GRASP_ZIP = f"{DRIVE_ROOT}/grasp.zip"
DRIVE_NB1_OUT  = f"{DRIVE_ROOT}/nb1_outputs"
DRIVE_NB7_OUT  = f"{DRIVE_ROOT}/grasp_transfer_outputs"

# Hyperparameters for Transfer Learning
SEQ_LEN = 72
GRASP_EPOCHS_FROZEN  = 40
GRASP_LR_FROZEN      = 3e-4
GRASP_EPOCHS_FULL    = 25
GRASP_LR_FULL        = 5e-5
GRASP_EPOCHS_SCRATCH = 50
GRASP_LR_SCRATCH     = 3e-4
FEW_SHOT_FRACS       = [0.1, 0.5, 1.0]
HIGH_FLUX_PERCENTILE = 90
# -------------------------------------------------------

# 1. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

assert torch.cuda.is_available(), "Please enable GPU: Runtime -> Change runtime type -> GPU"
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# 2. Setup Working Directory
WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib")

# 3. Unpack Code
print("\n[1/4] Unpacking Code...")
with zipfile.ZipFile(DRIVE_CODE_ZIP, "r") as z:
    z.extractall(WORK / "_code")
code_root = list((WORK / "_code").rglob("run_training.py"))[0].parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)

# 4. Unpack Datasets (GOES, OMNI, GRASP)
print("[2/4] Unpacking Datasets (OMNI & GRASP)...")
dst_omni = WORK / "datasets" / "omni"
dst_grasp = WORK / "datasets" / "grasp"

if not dst_omni.exists():
    with zipfile.ZipFile(DRIVE_DATA_ZIP, "r") as z:
        z.extractall(WORK / "_data")
    o = next((p for p in (WORK / "_data").rglob("omni") if p.is_dir()), None)
    shutil.copytree(o, dst_omni)

if not dst_grasp.exists():
    with zipfile.ZipFile(DRIVE_GRASP_ZIP, "r") as z:
        z.extractall(WORK / "_grasp")
    g = next((p for p in (WORK / "_grasp").rglob("grasp") if p.is_dir()), None) or (WORK / "_grasp")
    shutil.copytree(g, dst_grasp)

# 5. Restore Pre-trained Checkpoints
print("[3/4] Restoring Pre-trained Checkpoints...")
Path("checkpoints").mkdir(exist_ok=True)
for p in Path(DRIVE_NB1_OUT).rglob("seed_*"):
    if not p.is_dir(): continue
    if not (any(p.glob("*_best.pt")) or any(p.glob("*_best.zip"))): continue
    label = p.parent.name
    dest_p = WORK / "checkpoints" / label / p.name
    if not dest_p.exists():
        shutil.copytree(p, dest_p)

# 6. Imports & Dataloaders
from src.data.cdf_reader import read_wind_directory, read_grasp_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import VanillaTransformer
from src.model.storm_physnet import STORMPhysNet

print("[4/4] Preparing GRASP + OMNI Dataloaders...")
wind = read_wind_directory("datasets/omni")
grasp_flux_raw = read_grasp_directory("datasets/grasp")
# Resample GRASP 5-min flux to hourly mean to align with OMNI
grasp_flux_h = grasp_flux_raw.resample("1h").mean()
grasp_raw = grasp_flux_h.join(wind, how="inner")

grasp_pre = Preprocessor()
g_train, g_val, g_test = grasp_pre.fit_transform(grasp_raw)

grasp_train_loader, grasp_val_loader, grasp_test_loader = make_dataloaders(
    g_train, g_val, g_test,
    seq_len=SEQ_LEN, batch_size=64, storm_weight=10.0, num_workers=0
)
n_sw = grasp_test_loader.dataset.n_sw_features

# --- Helper Functions ---
def pe(yt, yp, yb):
    mse_p = mean_squared_error(yt, yp)
    mse_b = mean_squared_error(yt, yb)
    return 0.0 if mse_b == 0 else float(1.0 - mse_p / mse_b)

def build_storm_bz(n_sw_feat):
    return STORMPhysNet(
        n_sw_features=n_sw_feat, seq_len=SEQ_LEN, d_model=128, n_heads=4,
        n_transformer_layers=2, n_ssm_layers=2, d_state=64, d_ff=256,
        hidden_dim=64, n_horizons=3, dropout=0.1, ablation="none",
        backbone="transformer", gate_type="bz", use_spectral_head=False,
    )

def build_transformer(n_sw_feat):
    return VanillaTransformer(n_sw_features=n_sw_feat, seq_len=SEQ_LEN, n_horizons=3)

def load_ckpt(model, path):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # Filter out input layer shape mismatches for transfer learning
    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state.items():
        if k in model_state and v.shape != model_state[k].shape:
            pass # Drop mismatched layers (e.g. input projection)
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

def print_metrics(yt, yp, yb, st, label):
    thr = np.percentile(yt[:, 1], HIGH_FLUX_PERCENTILE)
    hi  = yt[:, 1] >= thr
    pe_45m = pe(yt[:, 0], yp[:, 0], yb[:, 0])
    pe_6h  = pe(yt[:, 1], yp[:, 1], yb[:, 1])
    pe_12h = pe(yt[:, 2], yp[:, 2], yb[:, 2])
    pe_st  = pe(yt[st, 1], yp[st, 1], yb[st, 1]) if st.any() else float("nan")
    pe_hi  = pe(yt[hi, 1], yp[hi, 1], yb[hi, 1]) if hi.any() else float("nan")
    
    def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
    print(f"{label:<25} | 45m: {fmt(pe_45m):>6} | 6h: {fmt(pe_6h):>6} | 12h: {fmt(pe_12h):>6} | Storm 6h: {fmt(pe_st):>6} | HiFlux: {fmt(pe_hi):>6}")

def fine_tune(model, train_loader, val_loader, epochs, lr, freeze_encoder=False):
    if freeze_encoder:
        for name, p in model.named_parameters():
            if any(k in name for k in ["flux_head", "storm_head", "var_head", "input_proj", "prop_delay.tau_cond_net"]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
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
            if not torch.isnan(loss):
                loss.backward(); opt.step()
                
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
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
    if best_state: model.load_state_dict(best_state)
    return model

def get_ckpt(label):
    d = Path(f"checkpoints/{label}/seed_42")
    cands = sorted(d.glob("*_best.pt")) + sorted(d.glob("*_best.zip"))
    return cands[0] if cands else None

# ======================================================================
# RUN TABLE II EXPERIMENTS
# ======================================================================
print("\n" + "="*72)
print("TABLE II EXPERIMENTS: GRASP DOMAIN TRANSFER")
print("="*72)

# A. Zero-shot Transformer
ckpt_tf = get_ckpt("transformer")
if ckpt_tf:
    model = load_ckpt(build_transformer(n_sw).to(device), ckpt_tf)
    yt, yp, yb, st = predict(model, grasp_test_loader)
    print_metrics(yt, yp, yb, st, "Zero-shot Transformer")

# B. Zero-shot STORM-BzGate
ckpt_bz = get_ckpt("storm_bz")
if ckpt_bz:
    model = load_ckpt(build_storm_bz(n_sw).to(device), ckpt_bz)
    yt, yp, yb, st = predict(model, grasp_test_loader)
    print_metrics(yt, yp, yb, st, "Zero-shot STORM-BzGate")

    # C. Frozen TL
    print("\nTraining Frozen TL (Encoder frozen)...")
    model = load_ckpt(build_storm_bz(n_sw).to(device), ckpt_bz)
    model = fine_tune(model, grasp_train_loader, grasp_val_loader, GRASP_EPOCHS_FROZEN, GRASP_LR_FROZEN, freeze_encoder=True)
    yt, yp, yb, st = predict(model, grasp_test_loader)
    print_metrics(yt, yp, yb, st, "Frozen TL")

    # D. Full TL
    print("\nTraining Full TL (All layers unlocked)...")
    model = load_ckpt(build_storm_bz(n_sw).to(device), ckpt_bz)
    model = fine_tune(model, grasp_train_loader, grasp_val_loader, GRASP_EPOCHS_FULL, GRASP_LR_FULL, freeze_encoder=False)
    yt, yp, yb, st = predict(model, grasp_test_loader)
    print_metrics(yt, yp, yb, st, "Full TL")

    # E. Scratch
    print("\nTraining from Scratch (No pre-training)...")
    model = build_storm_bz(n_sw).to(device)
    model = fine_tune(model, grasp_train_loader, grasp_val_loader, GRASP_EPOCHS_SCRATCH, GRASP_LR_SCRATCH, freeze_encoder=False)
    yt, yp, yb, st = predict(model, grasp_test_loader)
    print_metrics(yt, yp, yb, st, "Scratch")

    # F. Few-shot
    for frac in FEW_SHOT_FRACS:
        print(f"\nTraining Few-shot ({int(frac*100)}% data, Frozen TL)...")
        n_samples = max(1, int(len(grasp_train_loader.dataset) * frac))
        idx = np.random.choice(len(grasp_train_loader.dataset), n_samples, replace=False)
        subset_loader = DataLoader(Subset(grasp_train_loader.dataset, idx), batch_size=64, shuffle=True)
        
        model = load_ckpt(build_storm_bz(n_sw).to(device), ckpt_bz)
        model = fine_tune(model, subset_loader, grasp_val_loader, GRASP_EPOCHS_FROZEN, GRASP_LR_FROZEN, freeze_encoder=True)
        yt, yp, yb, st = predict(model, grasp_test_loader)
        print_metrics(yt, yp, yb, st, f"Few-shot {int(frac*100)}%")

print("\nGRASP Transfer Learning Complete.")

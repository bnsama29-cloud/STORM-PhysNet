# ============================================================
# WEEK 3 — Noise Robustness Evaluation
# Goal: Test if STORM's physics constraints provide robustness 
# against sensor noise (e.g., ACE/DSCOVR measurement errors).
# ============================================================
import os, glob, json, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/storm_physnet")
except ImportError:
    DRIVE = Path(r"f:\Downloads\ieee_final_fixed\drive\storm_physnet")

# -------------------- PATHS (edit if needed) --------------------
TF_ROOTS = [
    DRIVE / "nb1_stats_outputs" / "checkpoints" / "transformer",
    DRIVE / "tier1_extra_seeds" / "checkpoints" / "transformer",
]
STORM_ROOTS = [
    DRIVE / "nb1_stats_outputs" / "checkpoints" / "storm_bz",
    DRIVE / "tier1_extra_seeds" / "checkpoints" / "storm_bz",
]

OUT = DRIVE / "week3_robustness"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

SEEDS = list(range(42, 57))
# Noise levels to test (std of Gaussian noise added to standardized features)
NOISE_LEVELS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]
HORIZON_NAMES = ["45min", "6h", "12h"]
H_IDX = {"45min": 0, "6h": 1, "12h": 2}
# ----------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

# ---- setup code on /content ----
import sys
import zipfile
import shutil

if "google.colab" in sys.modules:
    os.chdir("/content")
    work = Path("/content/storm_w3")
    work.mkdir(exist_ok=True)
    os.chdir(work)

    # 1. Unpack code zip and setup src
    code_zip = DRIVE / "ieee_final_fixed.zip"
    unpacked_dir = Path("/content/storm_w3/_code")
    if code_zip.exists():
        unpacked_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'unzip -q -o "{code_zip}" -d "{unpacked_dir}"')

    hits = list(unpacked_dir.rglob("run_training.py"))
    if not hits: hits = list(DRIVE.rglob("run_training.py"))
    assert hits, "Cannot find run_training.py under Drive or zip."
    code_root = hits[0].parent

    for name in ["src", "configs"]:
        s, d = code_root / name, Path(name)
        if s.exists():
            if d.exists(): shutil.rmtree(d)
            shutil.copytree(s, d)

    if (code_root / "run_training.py").exists():
        shutil.copy2(code_root / "run_training.py", "run_training.py")

    for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
              "src/train/__init__.py"]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).touch()

    # 2. Extract Data
    data_zip = DRIVE / "datasets.zip"
    if data_zip.exists() and not Path("datasets").exists():
        os.system(f'unzip -q -o "{data_zip}" -d "_data"')
        for key in ["goes", "omni"]:
            src = next(p for p in Path("_data").rglob(key) if p.is_dir())
            dst = Path("datasets") / key
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(src, dst)

    os.system("pip -q install cdflib pandas scikit-learn pyyaml scipy matplotlib")
else:
    work = Path(r"f:\Downloads\ieee_final_fixed")
    os.chdir(work)

if str(work) not in sys.path:
    sys.path.insert(0, str(work))
for key in list(sys.modules.keys()):
    if key == "src" or key.startswith("src."): del sys.modules[key]

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders

# ---- metrics ----
def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    var = np.var(y)
    if var < 1e-12: return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / var)

def pe_pers(y, yhat, ypers):
    y, yhat, ypers = np.asarray(y).ravel(), np.asarray(yhat).ravel(), np.asarray(ypers).ravel()
    mse_pers = np.mean((y - ypers) ** 2)
    if mse_pers < 1e-12: return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / mse_pers)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y).ravel() - np.asarray(yhat).ravel()) ** 2)))

cfg = yaml.safe_load(open("configs/config.yaml"))
raw_goes = read_goes_directory("datasets/goes")
raw_wind = read_wind_directory("datasets/omni")
raw = raw_goes.join(raw_wind, how="inner").dropna()

pp_candidates = []
for r in TF_ROOTS + STORM_ROOTS:
    if r.exists(): pp_candidates.extend(list(r.rglob("preprocessor.pkl")))
assert pp_candidates, "No preprocessor.pkl found"
pp_path = pp_candidates[0]
print("preprocessor:", pp_path)

pre = Preprocessor.load(str(pp_path)) if hasattr(Preprocessor, "load") else pickle.load(open(pp_path, "rb"))
if not hasattr(pre, "year_split"): pre.year_split = None
if not hasattr(pre, "train_frac"): pre.train_frac = 0.70
if not hasattr(pre, "val_frac"): pre.val_frac = 0.15

assert hasattr(pre, "transform") and getattr(pre, "_fitted", True), \
    "Preprocessor not fitted/loadable — do not refit for Week3 paper numbers"
train_df, val_df, test_df = pre._split(raw)
train_df = pre.transform(train_df)
val_df   = pre.transform(val_df)
test_df  = pre.transform(test_df)

if hasattr(pre, "test_df"): test_df = pre.test_df

seq_len = cfg["data"].get("sequence_length", 72)
batch_size = cfg["data"].get("batch_size", 64)

try:
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, seq_len=seq_len, batch_size=batch_size
    )
except TypeError:
    try: train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, cfg)
    except TypeError: train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, cfg["data"])

batch = next(iter(test_loader))
n_sw = batch["x_sw"].size(-1) if isinstance(batch, dict) else batch[0].size(-1)

from src.model.baselines import VanillaTransformer as TFCls
from src.model.storm_physnet import STORMPhysNet

def build_tf(n_sw):
    for sig in [(n_sw, 72, 3), (n_sw, 72)]:
        try: return TFCls(n_sw_features=sig[0], seq_len=sig[1], n_horizons=sig[2] if len(sig)>2 else 3)
        except TypeError: pass
    return TFCls(n_sw, 72, 3)

def build_storm(n_sw):
    try: return STORMPhysNet(n_sw_features=n_sw, seq_len=72, n_horizons=3, gate_type="bz", backbone="transformer")
    except TypeError:
        try: return STORMPhysNet(n_sw_features=n_sw, seq_len=72)
        except TypeError: return STORMPhysNet(n_sw)

def load_state(model, path):
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]; break
        if any(str(k).startswith("module.") for k in ck.keys()):
            ck = {k.replace("module.", "", 1): v for k, v in ck.items()}
    model.load_state_dict(ck, strict=False)
    model.eval()
    return model

def find_ckpt(roots, seed: int):
    for root in roots:
        if not root.exists(): continue
        d = root / f"seed_{seed}"
        if not d.exists():
            hits = list(root.rglob(f"seed_{seed}"))
            if not hits: continue
            d = hits[0]
        pts = list(Path(d).rglob("*_best.pt")) + list(Path(d).rglob("*.pt"))
        pts = [p for p in pts if p.is_file() and p.stat().st_size > 1000]
        if pts: return pts[0]
    return None

@torch.no_grad()
def predict_loader(model, loader, noise_std=0.0):
    ys, yh, yp, st = [], [], [], []
    for batch in loader:
        if isinstance(batch, dict):
            x_sw = batch["x_sw"].to(device)
            x_flux = batch.get("x_flux", batch.get("flux_hist"))
            if x_flux is None: x_flux = batch.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device))
            else: x_flux = x_flux.to(device)
            y = batch["y_flux"].to(device)
            y_pers = batch.get("y_persist", batch.get("y_pers"))
            if y_pers is None: y_pers = y * 0
            else: y_pers = y_pers.to(device)
            storm = batch.get("storm_flag", batch.get("y_storm"))
            if storm is None: storm = torch.zeros(y.size(0), 1, device=device)
            else: storm = storm.to(device)
        else:
            x_sw = batch[0].to(device)
            x_flux = batch[1].to(device) if len(batch) > 1 else torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)
            y = batch[2].to(device)
            y_pers = batch[3].to(device) if len(batch) > 3 else torch.zeros_like(y)
            storm = batch[4].to(device) if len(batch) > 4 else torch.zeros(y.size(0), 1, device=device)

        # ---- ADD NOISE HERE ----
        if noise_std > 0.0:
            noise = torch.randn_like(x_sw) * noise_std
            x_sw = x_sw + noise

        try: out = model(x_sw, x_flux, y_pers)
        except TypeError:
            try: out = model(x_sw, x_flux)
            except TypeError: out = model(x_sw)

        if isinstance(out, dict):
            pred = out.get("flux_pred", out.get("pred", None))
            if pred is None:
                for v in out.values():
                    if torch.is_tensor(v) and v.dim() == 2 and v.size(1) == y.size(1):
                        pred = v; break
        else: pred = out
        ys.append(y.cpu().numpy())
        yh.append(pred.detach().cpu().numpy())
        yp.append(y_pers.cpu().numpy())
        st.append(storm.cpu().numpy())
    return (
        np.concatenate(ys, 0), np.concatenate(yh, 0),
        np.concatenate(yp, 0), np.concatenate(st, 0).ravel() > 0.5,
    )

def metrics_block(y, yhat, y_pers, storm):
    row = {}
    for name, j in H_IDX.items():
        row[f"PE_{name}"] = pe_clim(y[:, j], yhat[:, j])
        row[f"PE_pers_{name}"] = pe_pers(y[:, j], yhat[:, j], y_pers[:, j])
        if storm is not None and storm.any():
            row[f"PE_storm_{name}"] = pe_clim(y[storm, j], yhat[storm, j])
        else:
            row[f"PE_storm_{name}"] = float("nan")
    return row

noise_rows = []

for seed in SEEDS:
    p_tf = find_ckpt(TF_ROOTS, seed)
    p_st = find_ckpt(STORM_ROOTS, seed)
    print(f"\nseed {seed}: TF={p_tf}  STORM={p_st}")
    if p_tf is None or p_st is None:
        print("  SKIP missing ckpt")
        continue

    tf = load_state(build_tf(n_sw).to(device), p_tf)
    st = load_state(build_storm(n_sw).to(device), p_st)

    for noise in NOISE_LEVELS:
        y, y_tf, y_pers, storm = predict_loader(tf, test_loader, noise_std=noise)
        _, y_st, _, _ = predict_loader(st, test_loader, noise_std=noise)
        
        # Hybrid logic
        y_hyb = np.copy(y_tf)
        y_hyb[:, 0] = y_st[:, 0]
        
        # Ensemble logic (alpha = 0.3 from Week 2A)
        alpha = 0.3
        y_ens = alpha * y_st + (1.0 - alpha) * y_tf

        r_tf = metrics_block(y, y_tf, y_pers, storm); r_tf.update({"seed": seed, "system": "transformer", "noise": noise})
        r_st = metrics_block(y, y_st, y_pers, storm); r_st.update({"seed": seed, "system": "storm_bz", "noise": noise})
        r_hyb = metrics_block(y, y_hyb, y_pers, storm); r_hyb.update({"seed": seed, "system": "hybrid", "noise": noise})
        r_ens = metrics_block(y, y_ens, y_pers, storm); r_ens.update({"seed": seed, "system": "ensemble_0.3", "noise": noise})
        noise_rows.extend([r_tf, r_st, r_hyb, r_ens])

df = pd.DataFrame(noise_rows)
if not df.empty:
    df.to_csv(OUT / "tables" / "noise_robustness_raw.csv", index=False)
    
    metrics = [c for c in df.columns if c.startswith("PE_")]
    g = df.groupby(["system", "noise"])[metrics].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g.to_csv(OUT / "tables" / "noise_robustness_summary.csv", index=False)
    
    print("\n======== NOISE ROBUSTNESS ========")
    print(g[["system", "noise", "PE_45min_mean", "PE_6h_mean", "PE_pers_45min_mean"]].to_string(index=False))

    # ---- PLOTTING ----
    try:
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        pub_labels = {"storm_bz": "STORM-Bz", "transformer": "Transformer", "hybrid": "Hybrid", "ensemble_0.3": "Ensemble (α=0.3)"}
        for sys_name in ["storm_bz", "transformer", "hybrid", "ensemble_0.3"]:
            sub = g[g["system"] == sys_name]
            if sub.empty: continue
            
            axes[0].plot(sub["noise"], sub["PE_45min_mean"], marker="o", label=pub_labels.get(sys_name, sys_name), linewidth=2)
            axes[1].plot(sub["noise"], sub["PE_6h_mean"], marker="s", label=pub_labels.get(sys_name, sys_name), linewidth=2)
            
        axes[0].set_title("Robustness: PE (45 min) vs Sensor Noise")
        axes[1].set_title("Robustness: PE (6 h) vs Sensor Noise")
        
        for ax in axes:
            ax.set_xlabel("Noise Std (Standardized)")
            ax.set_ylabel("PE")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(OUT / "plots" / "noise_robustness.png", dpi=300)
        print("\nPlot saved to:", OUT / "plots" / "noise_robustness.png")
    except Exception as e:
        print("Could not generate plot:", e)

print("DONE →", OUT)

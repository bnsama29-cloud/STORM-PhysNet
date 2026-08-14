# ============================================================
# OPTIONAL EXTRAS: Bagged STORM Bootstrap CI & Interpretability
# ============================================================
import os, sys, shutil, yaml, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

WORK = Path("/content/storm_paper_optional")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

try: import cdflib
except ImportError: os.system("pip -q install cdflib pandas scikit-learn pyyaml scipy tqdm matplotlib")

# Set up code and data
DRIVE = Path("/content/drive/MyDrive/storm_physnet")
DRIVE_CODE_ZIP = DRIVE / "ieee_final_fixed.zip"
DRIVE_DATA_ZIP = DRIVE / "datasets.zip"
OUT = DRIVE / "paper_extra_stats"

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

for ds in ["goes", "omni"]:
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

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.model.storm_physnet import STORMPhysNet

def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    v = np.var(y)
    return 0.0 if v < 1e-12 else float(1 - np.mean((y - yhat)**2) / v)

cfg = yaml.safe_load(open("configs/config.yaml"))
raw = read_goes_directory("datasets/goes").join(read_wind_directory("datasets/omni"), how="inner")

pre_path = list(DRIVE.rglob("nb1_stats_outputs/**/preprocessor.pkl"))
if not pre_path: pre_path = list(DRIVE.rglob("**/storm_bz/**/preprocessor.pkl"))
pp = pre_path[0]

if hasattr(Preprocessor, "load"):
    pre = Preprocessor.load(str(pp))
else:
    pre = pickle.load(open(pp, "rb"))

if not hasattr(pre, "year_split"): pre.year_split = None
if not hasattr(pre, "train_frac"): pre.train_frac = 0.70
if not hasattr(pre, "val_frac"): pre.val_frac = 0.15

try:
    tr, va, te = pre._split(raw)
    tr, va, te = pre.transform(tr), pre.transform(va), pre.transform(te)
except Exception:
    tr, va, te = pre.fit_transform(raw)

from torch.utils.data import DataLoader
from src.data.dataloader import FluxDataset
test_ds = FluxDataset(te, seq_len=72)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
sw_cols = test_ds.sw_cols
batch = next(iter(test_loader))
n_sw = batch["x_sw"].shape[-1] if isinstance(batch, dict) else batch[0].shape[-1]

def build_st():
    try: return STORMPhysNet(n_sw_features=n_sw, seq_len=72, gate_type="bz", backbone="transformer")
    except TypeError: return STORMPhysNet(n_sw_features=n_sw, seq_len=72)

def load_ckpt(model, path):
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ck and isinstance(ck[k], dict): ck = ck[k]; break
        ck = {k.replace("module.", ""): v for k, v in ck.items()}
    model.load_state_dict(ck, strict=False); model.eval(); return model

def find_pt(label, seed):
    for root in [DRIVE/"nb1_stats_outputs"/"checkpoints"/label, DRIVE/"tier1_extra_seeds"/"checkpoints"/label]:
        d = root / f"seed_{seed}"
        if not d.exists(): continue
        pts = list(d.rglob("*_best.pt")) + list(d.rglob("*.pt"))
        pts = [p for p in pts if p.stat().st_size > 1000]
        if pts: return pts[0]
    return None

SEEDS = list(range(42, 57))
all_y_st = []
y_true, storm_true = None, None

print("\n" + "="*60)
print("1. BAGGED STORM: Bootstrap Confidence Intervals (2,000 resamples)")
print("="*60)

for seed in SEEDS:
    pst = find_pt("storm_bz", seed)
    if not pst: continue
    
    print(f"Loading STORM seed {seed}...")
    st = load_ckpt(build_st().to(device), pst)
    
    ys, yh, stm = [], [], []
    with torch.no_grad():
        for b in test_loader:
            x_sw = b["x_sw"].to(device)
            x_flux = b.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)).to(device)
            y = b["y_flux"].to(device)
            storm = b.get("storm_flag", torch.zeros(y.size(0), 1, device=device)).to(device)
            y_pers = b.get("y_persist", torch.zeros_like(y)).to(device)
            
            try: out = st(x_sw, x_flux, y_pers)
            except TypeError:
                try: out = st(x_sw, x_flux)
                except TypeError: out = st(x_sw)
            
            pred = out["flux_pred"] if isinstance(out, dict) else out
            ys.append(y.cpu().numpy()); yh.append(pred.cpu().numpy())
            stm.append(storm.cpu().numpy().ravel() > 0.5)
            
    if y_true is None:
        y_true = np.concatenate(ys)
        storm_true = np.concatenate(stm)
    all_y_st.append(np.concatenate(yh))

if all_y_st:
    rng = np.random.default_rng(0)
    N_seeds = len(all_y_st)
    n_boot = 2000
    
    results = {"PE_45min": [], "PE_6h": [], "PE_12h": [], "PE_storm_6h": []}
    
    for _ in range(n_boot):
        # Sample seeds with replacement
        idx = rng.integers(0, N_seeds, N_seeds)
        sampled_preds = np.array([all_y_st[i] for i in idx])
        bag_pred = np.mean(sampled_preds, axis=0)
        
        results["PE_45min"].append(pe_clim(y_true[:, 0], bag_pred[:, 0]))
        results["PE_6h"].append(pe_clim(y_true[:, 1], bag_pred[:, 1]))
        results["PE_12h"].append(pe_clim(y_true[:, 2], bag_pred[:, 2]))
        if storm_true.any():
            results["PE_storm_6h"].append(pe_clim(y_true[storm_true, 1], bag_pred[storm_true, 1]))
            
    summary = []
    for m in results:
        vals = np.array(results[m])
        mu = float(np.mean(vals))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        summary.append({"metric": m, "mean": mu, "ci95_lo": lo, "ci95_hi": hi, "n_seeds": N_seeds, "n_boot": n_boot})
        
    df_bagged = pd.DataFrame(summary)
    df_bagged.to_csv(OUT / "tables" / "bagged_bootstrap_ci.csv", index=False)
    print(df_bagged.to_string(index=False))


print("\n" + "="*60)
print("2. INTERPRETABILITY: Extracting Physics-Gate & Propagation Delay (Tau)")
print("="*60)

pst = find_pt("storm_bz", SEEDS[0])
if pst:
    st = load_ckpt(build_st().to(device), pst)
    taus, gates, bz_vals, vsw_vals = [], [], [], []
    
    bz_idx = sw_cols.index("bz") if "bz" in sw_cols else None
    vsw_idx = sw_cols.index("vsw") if "vsw" in sw_cols else None
    
    with torch.no_grad():
        for b in test_loader:
            x_sw = b["x_sw"].to(device)
            x_flux = b.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)).to(device)
            y = b["y_flux"].to(device)
            y_pers = b.get("y_persist", torch.zeros_like(y)).to(device)
            
            try: out = st(x_sw, x_flux, y_pers)
            except TypeError:
                try: out = st(x_sw, x_flux)
                except TypeError: out = st(x_sw)
                
            if isinstance(out, dict) and "gate_values" in out and "tau" in out:
                gate = out["gate_values"].cpu().numpy().ravel()
                tau = out["tau"].cpu().numpy().ravel()
                gates.extend(gate)
                taus.extend(tau)
                
                if bz_idx is not None:
                    bz_vals.extend(x_sw[:, -1, bz_idx].cpu().numpy().ravel())
                if vsw_idx is not None:
                    vsw_vals.extend(x_sw[:, -1, vsw_idx].cpu().numpy().ravel())

    if gates and taus:
        print("Data extracted successfully. Plotting scatter plots...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if bz_idx is not None:
            # Randomly sub-sample 2000 points so the scatter plot isn't too dense
            idx = np.random.choice(len(bz_vals), min(2000, len(bz_vals)), replace=False)
            bzv = np.array(bz_vals)[idx]
            gv = np.array(gates)[idx]
            
            axes[0].scatter(bzv, gv, alpha=0.3, color="blue", s=10)
            axes[0].set_xlabel("Bz (Solar Wind Magnetic Field)")
            axes[0].set_ylabel("Physics Gate Activation")
            axes[0].set_title("Interpretability: Physics Gate vs Bz")
            axes[0].grid(True, alpha=0.3)
            
        if vsw_idx is not None:
            idx = np.random.choice(len(vsw_vals), min(2000, len(vsw_vals)), replace=False)
            vswv = np.array(vsw_vals)[idx]
            tv = np.array(taus)[idx]
            
            axes[1].scatter(vswv, tv, alpha=0.3, color="red", s=10)
            axes[1].set_xlabel("Vsw (Solar Wind Speed)")
            axes[1].set_ylabel("Learned Propagation Delay (Tau)")
            axes[1].set_title("Interpretability: Learned Delay vs Vsw")
            axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(OUT / "figures" / "interpretability_scatter.png", dpi=300)
        plt.close()
        print("Saved interpretability scatter plot to: paper_extra_stats/figures/interpretability_scatter.png")
    else:
        print("Model does not return 'gate_values' or 'tau'. Skipping interpretability plot.")

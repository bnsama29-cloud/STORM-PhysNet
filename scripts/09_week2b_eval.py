# ============================================================
# WEEK 2B — Evaluation of Horizon-Conditioned STORM
# Goal: Compare storm_hz_cond vs storm_bz vs transformer
# ============================================================
import os, sys, shutil, yaml, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

WORK = Path("/content/storm_eval_2b")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

try: import cdflib
except ImportError: os.system("pip -q install cdflib pandas scikit-learn pyyaml tqdm")

# Set up code and data
DRIVE = Path("/content/drive/MyDrive/storm_physnet")
DRIVE_CODE_ZIP = DRIVE / "ieee_final_fixed.zip"
DRIVE_DATA_ZIP = DRIVE / "datasets.zip"

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
from src.data.dataloader import make_dataloaders

from src.model.baselines import VanillaTransformer as TFCls
from src.model.storm_physnet import STORMPhysNet

def build_tf(n_sw):
    for sig in [(n_sw, 72, 3), (n_sw, 72)]:
        try: return TFCls(n_sw_features=sig[0], seq_len=sig[1], n_horizons=sig[2] if len(sig)>2 else 3)
        except TypeError: pass
    return TFCls(n_sw, 72, 3)

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
raw_goes = read_goes_directory("datasets/goes")
raw_wind = read_wind_directory("datasets/omni")
raw = raw_goes.join(raw_wind, how="inner").dropna()

H_IDX = {"45min": 0, "6h": 1, "12h": 2}
def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    return float(1.0 - np.sum((y - yhat)**2) / (np.sum((y - np.mean(y))**2) + 1e-8))

def pe_pers(y, yhat, y_pers):
    y, yhat, y_pers = np.asarray(y).ravel(), np.asarray(yhat).ravel(), np.asarray(y_pers).ravel()
    return float(1.0 - np.sum((y - yhat)**2) / (np.sum((y - y_pers)**2) + 1e-8))

def predict_loader(model, loader):
    model.eval()
    preds, trues, pers = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_sw, x_flux, y_pers = batch["x_sw"].to(device), batch["x_flux"].to(device), batch["y_persist"].to(device)
            try: out = model(x_sw, x_flux, y_pers)
            except TypeError:
                try: out = model(x_sw, x_flux)
                except TypeError: out = model(x_sw)
                
            if isinstance(out, dict):
                pred = out.get("flux_pred", out.get("pred", None))
                if pred is None:
                    for v in out.values():
                        if torch.is_tensor(v) and v.dim() == 2 and v.size(1) == batch["y_flux"].size(1):
                            pred = v; break
            else: pred = out
            preds.append(pred.cpu().numpy())
            trues.append(batch["y_flux"].numpy())
            pers.append(batch["y_persist"].numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0), np.concatenate(pers, 0)

SEEDS = list(range(42, 57))
results = []

for seed in SEEDS:
    # 1. Find Transformer
    tf_pts = list(DRIVE.rglob(f"nb1_stats_outputs/checkpoints/transformer/seed_{seed}/*_best.pt")) + \
             list(DRIVE.rglob(f"tier1_extra_seeds/checkpoints/transformer/seed_{seed}/*_best.pt"))
    if not tf_pts: continue
    
    # 2. Find Base STORM (Week 1)
    st_pts = list(DRIVE.rglob(f"nb1_stats_outputs/checkpoints/storm_bz/seed_{seed}/*_best.pt")) + \
             list(DRIVE.rglob(f"tier1_extra_seeds/checkpoints/storm_bz/seed_{seed}/*_best.pt"))
    if not st_pts: continue
        
    # 3. Find Horizon Conditioned STORM (Week 2B)
    hz_pts = list(DRIVE.rglob(f"week2b_outputs/checkpoints/storm_hz_cond/seed_{seed}/*_best.pt"))
    if not hz_pts: continue
    
    pre_path = tf_pts[0].parent / "preprocessor.pkl"
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
    
    storm = test_df["storm_flag"].values[72:] if "storm_flag" in test_df.columns else None
    if storm is not None and len(storm) > len(test_ds): storm = storm[:len(test_ds)]
    storm = storm > 0 if storm is not None else None

    batch = next(iter(test_loader))
    n_sw = batch["x_sw"].size(-1) if isinstance(batch, dict) else batch[0].size(-1)

    print(f"\n[{seed}] Evaluating TF, STORM_Base, STORM_Hz...")
    
    m_tf = load_state(build_tf(n_sw).to(device), tf_pts[0])
    m_bz = load_state(build_storm(n_sw).to(device), st_pts[0])
    m_hz = load_state(build_storm(n_sw).to(device), hz_pts[0])

    y_tf, y, y_pers = predict_loader(m_tf, test_loader)
    y_bz, _, _      = predict_loader(m_bz, test_loader)
    y_hz, _, _      = predict_loader(m_hz, test_loader)
    
    def add_res(sys_name, yhat):
        row = {"seed": seed, "system": sys_name}
        for name, j in H_IDX.items():
            row[f"PE_{name}"] = pe_clim(y[:, j], yhat[:, j])
            row[f"PE_pers_{name}"] = pe_pers(y[:, j], yhat[:, j], y_pers[:, j])
            if storm is not None and storm.any():
                row[f"PE_storm_{name}"] = pe_clim(y[storm, j], yhat[storm, j])
        results.append(row)

    add_res("transformer", y_tf)
    add_res("storm_bz (base)", y_bz)
    add_res("storm_hz (week2b)", y_hz)

print("\n\n======== WEEK 2B EVALUATION RESULT ========")
df = pd.DataFrame(results)
if not df.empty:
    out_csv = DRIVE / "week2b_outputs" / "week2b_evaluation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}\n")
    
    g = df.groupby("system").mean(numeric_only=True).round(3)
    cols = ["PE_45min", "PE_pers_45min", "PE_6h", "PE_storm_6h", "PE_12h"]
    exist = [c for c in cols if c in g.columns]
    print(g[exist].to_markdown())
else:
    print("No complete seeds found for Week 2B yet!")

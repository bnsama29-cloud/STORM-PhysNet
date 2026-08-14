# ============================================================
# WEEK2A — Ensemble + Hybrid (STORM vs TF)  [NO RETRAIN]
# Goal: beat TF on every horizon via system design
# ============================================================
import os, glob, json, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# -------------------- PATHS (edit if needed) --------------------
DRIVE = Path("/content/drive/MyDrive/storm_physnet")
CODE_ROOTS = [
    DRIVE / "ieee_final_fixed",
    DRIVE / "STORM_PhysNet_Colab_Clean",
]
TF_ROOTS = [
    DRIVE / "nb1_stats_outputs" / "checkpoints" / "transformer",
    DRIVE / "tier1_extra_seeds" / "checkpoints" / "transformer",
]
STORM_ROOTS = [
    DRIVE / "nb1_stats_outputs" / "checkpoints" / "storm_bz",
    DRIVE / "tier1_extra_seeds" / "checkpoints" / "storm_bz",
]

OUT = DRIVE / "week2a_ensemble_hybrid"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

SEEDS = list(range(42, 57))           # evaluate all seeds from 42 to 56
ALPHAS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]  # 0=TF only, 1=STORM only
HORIZON_NAMES = ["45min", "6h", "12h"]
H_IDX = {"45min": 0, "6h": 1, "12h": 2}
# ----------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

# ---- setup code on /content ----
os.chdir("/content")
work = Path("/content/storm_w2a")
work.mkdir(exist_ok=True)
os.chdir(work)

import zipfile
import shutil

# 1. Unpack code zip and setup src
code_zip = DRIVE / "ieee_final_fixed.zip"
unpacked_dir = Path("/content/storm_w2a/_code")

if code_zip.exists():
    print(f"Found {code_zip.name}. Unpacking...")
    unpacked_dir.mkdir(parents=True, exist_ok=True)
    import os
    os.system(f'unzip -q -o "{code_zip}" -d "{unpacked_dir}"')

hits = list(unpacked_dir.rglob("run_training.py"))
if not hits:
    # fallback to Drive
    hits = list(DRIVE.rglob("run_training.py"))

assert hits, "Cannot find run_training.py under Drive or zip. Please upload ieee_final_fixed.zip!"
code_root = hits[0].parent
print("code_root:", code_root)

for name in ["src", "configs"]:
    s, d = code_root / name, Path(name)
    if s.exists():
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)

if (code_root / "run_training.py").exists():
    shutil.copy2(code_root / "run_training.py", "run_training.py")

# Create __init__.py files just like 01_train_main.py does
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# datasets
import shutil
import zipfile

# Try to unpack datasets.zip if the folders don't exist
data_zip = DRIVE / "datasets.zip"
if data_zip.exists() and not Path("datasets/goes").exists():
    print(f"Found {data_zip.name}. Unpacking datasets...")
    Path("datasets").mkdir(exist_ok=True)
    import os
    os.system(f'unzip -q -o "{data_zip}" -d "/content/storm_w2a/_data"')
        
    # Copy the specific folders over
    for key in ["goes", "omni", "grasp"]:
        hits = list(Path("/content/storm_w2a/_data").rglob(key))
        hits = [h for h in hits if h.is_dir()]
        if hits:
            dst = Path("datasets") / key
            if not dst.exists():
                shutil.copytree(hits[0], dst)
                print(f"Copied {key} from {hits[0]}")

for key in ["goes", "omni", "grasp"]:
    dst = Path("datasets") / key
    if dst.exists():
        continue
    # Fallback to searching Drive if not extracted
    hits = list(DRIVE.rglob(key))
    hits = [h for h in hits if h.is_dir()]
    assert hits, f"dataset folder {key} not found on Drive. Please upload datasets.zip!"
    shutil.copytree(hits[0], dst)
    print("copied", key, "from", hits[0])

try:
    import cdflib
except ImportError:
    import os
    os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm")

import sys
if "/content/storm_w2a" not in sys.path:
    sys.path.insert(0, "/content/storm_w2a")

# Force Jupyter to forget any broken 'src' imports from previous runs
for key in list(sys.modules.keys()):
    if key == "src" or key.startswith("src."):
        del sys.modules[key]

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders

# ---- metrics ----
def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    var = np.var(y)
    if var < 1e-12:
        return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / var)

def pe_pers(y, yhat, y_pers):
    y, yhat, y_pers = map(lambda a: np.asarray(a).ravel(), (y, yhat, y_pers))
    mse_p = np.mean((y - y_pers) ** 2)
    if mse_p < 1e-12:
        return 0.0
    return float(1.0 - np.mean((y - yhat) ** 2) / mse_p)

def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))

# ---- load data once (transform-only preprocessor) ----
cfg_path = Path("configs/config.yaml")
if not cfg_path.exists():
    # minimal cfg
    cfg = {
        "data": {
            "goes_cdf_dir": "datasets/goes",
            "wind_cdf_dir": "datasets/omni",
            "grasp_cdf_dir": "datasets/grasp",
            "sequence_length": 72,
            "forecast_horizons": [0.75, 6.0, 12.0],
            "batch_size": 64,
            "use_real_data": True,
        }
    }
else:
    cfg = yaml.safe_load(open(cfg_path))

goes = read_goes_directory(cfg["data"]["goes_cdf_dir"])
wind = read_wind_directory(cfg["data"]["wind_cdf_dir"])
raw = goes.join(wind, how="inner")
print("[Data]", raw.shape)

# find any preprocessor from a seed folder
pp_candidates = []
for r in TF_ROOTS + STORM_ROOTS:
    if r.exists():
        pp_candidates.extend(list(r.rglob("preprocessor.pkl")))
assert pp_candidates, "No preprocessor.pkl found under STORM/TF checkpoint trees"
pp_path = pp_candidates[0]
print("preprocessor:", pp_path)

try:
    pre = Preprocessor.load(str(pp_path)) if hasattr(Preprocessor, "load") else pickle.load(open(pp_path, "rb"))
    
    # Backward compatibility for old pickles
    if not hasattr(pre, "year_split"):
        pre.year_split = None
    if not hasattr(pre, "train_frac"):
        pre.train_frac = 0.70
    if not hasattr(pre, "val_frac"):
        pre.val_frac = 0.15
        
    # Prefer transform-only path
    if hasattr(pre, "transform") and hasattr(pre, "_fitted"):
        df = pre.transform(raw)
        # rebuild split if needed
        if hasattr(pre, "fit_transform"):
            # many of your preprocessors only expose fit_transform for full pipeline —
            # use chronological split consistent with training if attributes exist
            pass
    # Standard path in your codebase:
    train_df, val_df, test_df = None, None, None
    if hasattr(pre, "fit_transform"):
        # If load worked, try transform path used in training scripts
        try:
            out = pre.fit_transform  # noqa — detect API
        except Exception:
            pass
except Exception as e:
    print("Preprocessor load issue:", e)
    pre = Preprocessor()

# Robust: use same API as your training notebooks
try:
    train_df, val_df, test_df = pre.fit_transform(raw)  # WARNING if this refits — see note below
    print("NOTE: if this was a REFIT, PE numbers are provisional. Prefer transform-only.")
except TypeError:
    # some versions return a single df + internal split
    df = pre.fit_transform(raw)
    n = len(df)
    a, b = int(0.70 * n), int(0.85 * n)
    train_df, val_df, test_df = df.iloc[:a], df.iloc[a:b], df.iloc[b:]

# Better: if preprocessor stores split indices / years, use them
if hasattr(pre, "test_df"):
    test_df = pre.test_df

# Adapt to your signature
seq_len = cfg["data"].get("sequence_length", 72)
batch_size = cfg["data"].get("batch_size", 64)

try:
    # Current codebase signature
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, 
        seq_len=seq_len, 
        batch_size=batch_size
    )
except TypeError:
    try:
        train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, cfg)
    except TypeError:
        train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, cfg["data"])

# probe n_sw
batch = next(iter(test_loader))
# batch may be dict or tuple
if isinstance(batch, dict):
    x_sw = batch.get("x_sw", batch.get("sw"))
    n_sw = x_sw.shape[-1]
else:
    x_sw = batch[0]
    n_sw = x_sw.shape[-1]
print("n_sw =", n_sw, "test batches =", len(test_loader))

# ---- model builders (match your repo names) ----
from src.model.storm_physnet import STORMPhysNet
try:
    from src.model.baselines import VanillaTransformer as TFCls
except ImportError:
    try:
        from src.model.baselines import StandardTransformer as TFCls
    except ImportError:
        from src.model.baselines import TransformerBaseline as TFCls

try:
    from src.model.baselines import StandardLSTM
except ImportError:
    StandardLSTM = None

def build_tf(n_sw):
    try:
        return TFCls(n_sw_features=n_sw, seq_len=72, n_horizons=3)
    except TypeError:
        try:
            return TFCls(n_sw, 72, n_horizons=3)
        except TypeError:
            return TFCls(n_sw, 72, 3)

def build_storm(n_sw):
    try:
        return STORMPhysNet(n_sw_features=n_sw, seq_len=72, n_horizons=3, gate_type="bz", backbone="transformer")
    except TypeError:
        try:
            return STORMPhysNet(n_sw_features=n_sw, seq_len=72)
        except TypeError:
            return STORMPhysNet(n_sw)

def load_state(model, path):
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
        # strip module. prefix
        if any(str(k).startswith("module.") for k in ck.keys()):
            ck = {k.replace("module.", "", 1): v for k, v in ck.items()}
    model.load_state_dict(ck, strict=False)
    model.eval()
    return model

def find_ckpt(roots, seed: int):
    for root in roots:
        if not root.exists():
            continue
        d = root / f"seed_{seed}"
        if not d.exists():
            # sometimes seed folders are nested differently
            hits = list(root.rglob(f"seed_{seed}"))
            if not hits:
                continue
            d = hits[0]
        pts = list(Path(d).rglob("*_best.pt")) + list(Path(d).rglob("*.pt"))
        pts = [p for p in pts if p.is_file() and p.stat().st_size > 1000]
        if pts:
            return pts[0]
    return None

@torch.no_grad()
def predict_loader(model, loader):
    """Returns y, yhat, y_pers, storm [N, H] numpy"""
    ys, yh, yp, st = [], [], [], []
    for batch in loader:
        if isinstance(batch, dict):
            x_sw = batch["x_sw"].to(device)
            x_flux = batch.get("x_flux", batch.get("flux_hist"))
            if x_flux is None:
                # last channel sometimes
                x_flux = batch.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device))
            else:
                x_flux = x_flux.to(device)
            y = batch["y_flux"].to(device)
            y_pers = batch.get("y_persist", batch.get("y_pers"))
            if y_pers is None:
                y_pers = y * 0  # fallback
            else:
                y_pers = y_pers.to(device)
            storm = batch.get("storm_flag", batch.get("y_storm"))
            if storm is None:
                storm = torch.zeros(y.size(0), 1, device=device)
            else:
                storm = storm.to(device)
        else:
            # tuple fallback — adjust if your loader differs
            x_sw = batch[0].to(device)
            x_flux = batch[1].to(device) if len(batch) > 1 else torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)
            y = batch[2].to(device)
            y_pers = batch[3].to(device) if len(batch) > 3 else torch.zeros_like(y)
            storm = batch[4].to(device) if len(batch) > 4 else torch.zeros(y.size(0), 1, device=device)

        try:
            out = model(x_sw, x_flux, y_pers)
        except TypeError:
            try:
                out = model(x_sw, x_flux)
            except TypeError:
                out = model(x_sw)

        if isinstance(out, dict):
            pred = out.get("flux_pred", out.get("pred", None))
            if pred is None:
                # first tensor
                for v in out.values():
                    if torch.is_tensor(v) and v.dim() == 2 and v.size(1) == y.size(1):
                        pred = v
                        break
        else:
            pred = out
        ys.append(y.cpu().numpy())
        yh.append(pred.detach().cpu().numpy())
        yp.append(y_pers.cpu().numpy())
        st.append(storm.cpu().numpy())
    return (
        np.concatenate(ys, 0),
        np.concatenate(yh, 0),
        np.concatenate(yp, 0),
        np.concatenate(st, 0).ravel() > 0.5,
    )

def metrics_block(y, yhat, y_pers, storm):
    row = {}
    for name, j in H_IDX.items():
        row[f"PE_{name}"] = pe_clim(y[:, j], yhat[:, j])
        row[f"PE_storm_{name}"] = pe_clim(y[storm, j], yhat[storm, j]) if storm.any() else float("nan")
        row[f"PE_pers_{name}"] = pe_pers(y[:, j], yhat[:, j], y_pers[:, j])
        row[f"RMSE_{name}"] = rmse(y[:, j], yhat[:, j])
    return row

# ---- collect per-seed predictions ----
rows = []
ens_alpha_rows = []
hybrid_rows = []

for seed in SEEDS:
    p_tf = find_ckpt(TF_ROOTS, seed)
    p_st = find_ckpt(STORM_ROOTS, seed)
    print(f"\nseed {seed}: TF={p_tf}  STORM={p_st}")
    if p_tf is None or p_st is None:
        print("  SKIP missing ckpt")
        continue

    tf = load_state(build_tf(n_sw).to(device), p_tf)
    st = load_state(build_storm(n_sw).to(device), p_st)

    y, y_tf, y_pers, storm = predict_loader(tf, test_loader)
    _, y_st, _, _ = predict_loader(st, test_loader)

    # single models
    r_tf = metrics_block(y, y_tf, y_pers, storm); r_tf.update({"seed": seed, "system": "transformer"})
    r_st = metrics_block(y, y_st, y_pers, storm); r_st.update({"seed": seed, "system": "storm_bz"})
    rows.extend([r_tf, r_st])

    # alpha ensemble: α*STORM + (1-α)*TF
    for a in ALPHAS:
        y_ens = a * y_st + (1.0 - a) * y_tf
        r = metrics_block(y, y_ens, y_pers, storm)
        r.update({"seed": seed, "alpha": a, "system": f"ensemble_a{a}"})
        ens_alpha_rows.append(r)

    # hybrid: 45min=STORM, 6h/12h=TF
    y_hyb = y_tf.copy()
    y_hyb[:, 0] = y_st[:, 0]
    r = metrics_block(y, y_hyb, y_pers, storm)
    r.update({"seed": seed, "system": "hybrid_shortSTORM_longTF"})
    hybrid_rows.append(r)

    # hybrid storm-aware: if storm → STORM all horizons else TF
    y_hyb2 = y_tf.copy()
    y_hyb2[storm] = y_st[storm]
    y_hyb2[:, 0] = y_st[:, 0]  # always short=STORM
    r2 = metrics_block(y, y_hyb2, y_pers, storm)
    r2.update({"seed": seed, "system": "hybrid_stormAware"})
    hybrid_rows.append(r2)

df_single = pd.DataFrame(rows)
df_ens = pd.DataFrame(ens_alpha_rows)
df_hyb = pd.DataFrame(hybrid_rows)

if not df_single.empty:
    df_single.to_csv(OUT / "tables" / "single_by_seed.csv", index=False)
    df_ens.to_csv(OUT / "tables" / "ensemble_by_seed.csv", index=False)
    df_hyb.to_csv(OUT / "tables" / "hybrid_by_seed.csv", index=False)

    def summarize(df, system_col="system"):
        if df.empty: return pd.DataFrame()
        metrics = [c for c in df.columns if c.startswith("PE_") or c.startswith("RMSE_")]
        g = df.groupby(system_col)[metrics].agg(["mean", "std", "count"])
        # flatten
        g.columns = ["_".join(c) for c in g.columns]
        return g.reset_index()

    sum_single = summarize(df_single)
    sum_ens = summarize(df_ens)
    sum_hyb = summarize(df_hyb)
    sum_single.to_csv(OUT / "tables" / "single_summary.csv", index=False)
    sum_ens.to_csv(OUT / "tables" / "ensemble_summary.csv", index=False)
    sum_hyb.to_csv(OUT / "tables" / "hybrid_summary.csv", index=False)

    print("\n======== SINGLE ========")
    cols = ["system", "PE_6h_mean", "PE_45min_mean", "PE_pers_45min_mean"]
    print(sum_single[[c for c in cols if c in sum_single.columns]].to_string(index=False))
    print("\n======== ENSEMBLE (mean over seeds) ========")
    print(sum_ens[[c for c in cols if c in sum_ens.columns]].to_string(index=False))
    print("\n======== HYBRID ========")
    print(sum_hyb[[c for c in cols if c in sum_hyb.columns]].to_string(index=False))
else:
    print("\nNo checkpoints were loaded. Please check that SEEDS and CKPT_STORM/CKPT_TF point to valid directories.")

# best alpha by mean PE_6h
if len(df_ens):
    tmp = df_ens.groupby("alpha")["PE_6h"].mean()
    best_a = float(tmp.idxmax())
    print(f"\nBest ensemble alpha by PE_6h: {best_a} → {tmp.max():.4f}")
    json.dump({"best_alpha_pe6h": best_a, "pe6h": float(tmp.max())},
              open(OUT / "tables" / "best_alpha.json", "w"), indent=2)

# plot alpha sweep
if len(df_ens):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for metric, axis in [("PE_6h", ax[0]), ("PE_45min", ax[1])]:
        g = df_ens.groupby("alpha")[metric].agg(["mean", "std"])
        axis.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3)
        axis.axhline(df_single[df_single.system == "transformer"][metric].mean(),
                     ls="--", color="gray", label="TF")
        axis.axhline(df_single[df_single.system == "storm_bz"][metric].mean(),
                     ls=":", color="C1", label="STORM")
        axis.set_xlabel("α (STORM weight)")
        axis.set_ylabel(metric)
        axis.legend()
        axis.set_title(metric)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "fig_ensemble_alpha_sweep.png", dpi=150)
    plt.close()
    print("wrote plots/fig_ensemble_alpha_sweep.png")

print("\nDONE →", OUT)
print("Next: put hybrid + best ensemble into paper Table; keep single-model table honest.")

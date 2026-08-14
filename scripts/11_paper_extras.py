# ============================================================
# P0 / P1 PAPER EXTRAS: Stats, Case Studies, and Ablations
# Combines P0-A, P0-B, and P1 priorities into a single run.
# ============================================================
import os, json, pickle, warnings, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/storm_physnet")
except ImportError:
    DRIVE = Path(r"f:\Downloads\ieee_final_fixed\drive\storm_physnet")
OUT = DRIVE / "paper_extra_stats"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

SEEDS = list(range(42, 57))
N_BOOT = 2000
RNG = np.random.default_rng(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

if "google.colab" in sys.modules:
    os.chdir("/content")
    work = Path("/content/storm_paper_extras"); work.mkdir(exist_ok=True); os.chdir(work)

    code_zip = DRIVE / "ieee_final_fixed.zip"
    os.system(f'unzip -q -o "{code_zip}" -d _code')
    code_root = next(Path("_code").rglob("run_training.py")).parent
    for name in ["src", "configs"]:
        s, d = code_root / name, Path(name)
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(s, d)
    os.system(f'unzip -q -o "{DRIVE / "datasets.zip"}" -d _data')
    for key in ["goes", "omni"]:
        src = next(p for p in Path("_data").rglob(key) if p.is_dir())
        dst = Path("datasets") / key
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst)
    os.system("pip -q install cdflib pandas scikit-learn pyyaml scipy tqdm matplotlib")
else:
    work = Path(r"f:\Downloads\ieee_final_fixed")
    os.chdir(work)
sys.path.insert(0, str(work))
for k in list(sys.modules):
    if k == "src" or k.startswith("src."): del sys.modules[k]

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.storm_physnet import STORMPhysNet
try:
    from src.model.baselines import VanillaTransformer as TFCls
except Exception:
    from src.model.baselines import StandardTransformer as TFCls

def pe_clim(y, yhat):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    v = np.var(y)
    return 0.0 if v < 1e-12 else float(1 - np.mean((y - yhat)**2) / v)

cfg = yaml.safe_load(open("configs/config.yaml"))
raw = read_goes_directory("datasets/goes").join(read_wind_directory("datasets/omni"), how="inner")
# preprocessor from any seed
pp = next(Path(DRIVE).rglob("nb1_stats_outputs/**/preprocessor.pkl"), None)
if pp is None:
    pp = next(Path(DRIVE).rglob("**/storm_bz/**/preprocessor.pkl"))
if hasattr(Preprocessor, "load"):
    pre = Preprocessor.load(str(pp))
else:
    pre = pickle.load(open(pp, "rb"))

# Backward compatibility for old pickles
if not hasattr(pre, "year_split"): pre.year_split = None
if not hasattr(pre, "train_frac"): pre.train_frac = 0.70
if not hasattr(pre, "val_frac"): pre.val_frac = 0.15
# Prefer transform-only; fall back carefully
try:
    tr, va, te = pre._split(raw)
    tr, va, te = pre.transform(tr), pre.transform(va), pre.transform(te)
except Exception:
    tr, va, te = pre.fit_transform(raw)
    print("WARNING: refit path used")

from torch.utils.data import DataLoader
from src.data.dataloader import FluxDataset
test_ds = FluxDataset(te, seq_len=72)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

batch = next(iter(test_loader))
n_sw = batch["x_sw"].shape[-1] if isinstance(batch, dict) else batch[0].shape[-1]
sw_cols = test_loader.dataset.sw_cols

def build_tf():
    try: return TFCls(n_sw_features=n_sw, seq_len=72, n_horizons=3)
    except TypeError: return TFCls(n_sw, 72, 3)

def build_st():
    try: return STORMPhysNet(n_sw_features=n_sw, seq_len=72, gate_type="bz", backbone="transformer")
    except TypeError: return STORMPhysNet(n_sw_features=n_sw, seq_len=72)

def load_ckpt(model, path):
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]; break
        ck = {k.replace("module.", ""): v for k, v in ck.items()}
    model.load_state_dict(ck, strict=False); model.eval(); return model

def find_pt(label, seed):
    for root in [DRIVE/"nb1_stats_outputs"/"checkpoints"/label,
                 DRIVE/"tier1_extra_seeds"/"checkpoints"/label]:
        d = root / f"seed_{seed}"
        if not d.exists(): continue
        pts = list(d.rglob("*_best.pt")) + list(d.rglob("*.pt"))
        pts = [p for p in pts if p.stat().st_size > 1000]
        if pts: return pts[0]
    return None

@torch.no_grad()
def predict(model, loader, dropout_idxs=None):
    ys, yh, st = [], [], []
    for batch in loader:
        x_sw = batch["x_sw"].to(device)
        if dropout_idxs is not None:
            x_sw[:, :, dropout_idxs] = 0.0 # Zero out missing features for P1 ablation
        x_flux = batch.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)).to(device)
        y = batch["y_flux"].to(device)
        storm = batch.get("storm_flag", torch.zeros(y.size(0), 1, device=device)).to(device)
        y_pers = batch.get("y_persist", torch.zeros_like(y)).to(device)
        try: out = model(x_sw, x_flux, y_pers)
        except TypeError:
            try: out = model(x_sw, x_flux)
            except TypeError: out = model(x_sw)
        pred = out["flux_pred"] if isinstance(out, dict) else out
        ys.append(y.cpu().numpy()); yh.append(pred.cpu().numpy())
        st.append(storm.cpu().numpy().ravel() > 0.5)
    return np.concatenate(ys), np.concatenate(yh), np.concatenate(st)


# ============================================================
# P0-A / P1: Bootstrap Statistics, Paired Tests, Bagging
# ============================================================
print("\n" + "="*60)
print("P0-A & P1: Bootstrap Statistics, T-Tests, and Bagging")
print("="*60)

rows = []
all_y_st = [] # For P1 Bagging

for seed in []:
    ptf, pst = find_pt("transformer", seed), find_pt("storm_bz", seed)
    if not ptf or not pst:
        print("skip", seed); continue
    tf = load_ckpt(build_tf().to(device), ptf)
    st = load_ckpt(build_st().to(device), pst)
    
    print(f"Evaluating Seed {seed}...")
    y, y_tf, storm = predict(tf, test_loader)
    _, y_st, _ = predict(st, test_loader)
    
    all_y_st.append(y_st)
    
    y_ens = 0.3 * y_st + 0.7 * y_tf
    y_hyb = y_tf.copy(); y_hyb[:, 0] = y_st[:, 0]
    for name, pred in [("transformer", y_tf), ("storm_bz", y_st),
                       ("ensemble_0.3", y_ens), ("hybrid", y_hyb)]:
        rows.append({
            "seed": seed, "system": name,
            "PE_45min": pe_clim(y[:, 0], pred[:, 0]),
            "PE_6h": pe_clim(y[:, 1], pred[:, 1]),
            "PE_12h": pe_clim(y[:, 2], pred[:, 2]),
            "PE_storm_6h": pe_clim(y[storm, 1], pred[storm, 1]) if storm.any() else np.nan,
        })

# P1: Multi-seed STORM bag (no TF)
if False:
    bagged_st = np.mean(all_y_st, axis=0)
    rows.append({
        "seed": "BAGGED", "system": "storm_bz_bagged",
        "PE_45min": pe_clim(y[:, 0], bagged_st[:, 0]),
        "PE_6h": pe_clim(y[:, 1], bagged_st[:, 1]),
        "PE_12h": pe_clim(y[:, 2], bagged_st[:, 2]),
        "PE_storm_6h": pe_clim(y[storm, 1], bagged_st[storm, 1]) if storm.any() else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "tables" / "per_seed_systems.csv", index=False)

def boot_ci(x, n=N_BOOT):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return float(np.mean(x)), np.nan, np.nan
    means = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(x)), float(lo), float(hi)

# Summary + CI
if not df.empty:
    sum_rows = []
    for sys_name in df.system.unique():
        sub = df[df.system == sys_name]
        for m in ["PE_45min", "PE_6h", "PE_12h", "PE_storm_6h"]:
            mu, lo, hi = boot_ci(sub[m].values)
            sum_rows.append({"system": sys_name, "metric": m, "mean": mu, "ci95_lo": lo, "ci95_hi": hi, "n": sub[m].notna().sum()})
    pd.DataFrame(sum_rows).to_csv(OUT / "tables" / "bootstrap_ci.csv", index=False)
    print("\n--- BOOTSTRAP CIs ---")
    print(pd.DataFrame(sum_rows).to_string(index=False))

# Paired tests vs transformer
if not df.empty:
    base = df[df.system == "transformer"].set_index("seed")
    tests = []
for sys_name in []:
    oth = df[df.system == sys_name].set_index("seed")
    for m in ["PE_45min", "PE_6h", "PE_storm_6h"]:
        a = base[m]; b = oth[m]
        idx = a.index.intersection(b.index)
        idx = [i for i in idx if i != "BAGGED"]
        d = (b.loc[idx] - a.loc[idx]).dropna()
        if len(d) < 3: continue
        t, p = stats.ttest_rel(b.loc[d.index], a.loc[d.index])
        tests.append({"system": sys_name, "metric": m, "delta_mean": float(d.mean()),
                      "t": float(t), "p": float(p), "n": int(len(d)),
                      "cohen_d": float(d.mean() / (d.std() + 1e-12))})
    pd.DataFrame(tests).to_csv(OUT / "tables" / "paired_vs_tf.csv", index=False)
    print("\n--- PAIRED T-TESTS vs TRANSFORMER ---")
    print(pd.DataFrame(tests).to_string(index=False))


# ============================================================
# P0-B: Case studies (quiet / enhancement / dropout)
# ============================================================
print("\n" + "="*60)
print("P0-B: Case Studies (Enhancement, Dropout, Quiet)")
print("="*60)

# Build a time-indexed test frame for event picking
if "log_flux" not in te.columns:
    for c in te.columns:
        if "flux" in c.lower():
            te.rename(columns={c: "log_flux"}, inplace=True); break

flux = te["log_flux"].values if "log_flux" in te.columns else te.iloc[:, 0].values
idx = np.arange(len(te))
enh = int(np.nanargmax(flux))

win = 24
drops = []
for i in range(win, len(flux) - win):
    drops.append((flux[i] - flux[i - win], i))
dropout = int(min(drops, key=lambda t: t[0])[1]) if drops else len(flux)//2

var = pd.Series(flux).rolling(48, center=True).std().fillna(1e9).values
quiet = int(np.nanargmin(var))

events = {"enhancement": enh, "dropout": dropout, "quiet": quiet}
print("event indices", events)

tf = load_ckpt(build_tf().to(device), find_pt("transformer", SEEDS[0]))
st = load_ckpt(build_st().to(device), find_pt("storm_bz", SEEDS[0]))

y, y_tf, _ = predict(tf, test_loader)
_, y_st, _ = predict(st, test_loader)
y_ens = 0.3 * y_st + 0.7 * y_tf

n = len(y)
half = 72

def panel(name, center):
    c = int(np.clip(center, half, n - half - 1))
    sl = slice(c - half, c + half)
    t = np.arange(sl.start, sl.stop)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, h, title in zip(axes, [0, 1, 2], ["45 min", "6 h", "12 h"]):
        ax.plot(t, y[sl, h], "k", lw=2, label="True")
        ax.plot(t, y_tf[sl, h], label="Transformer", alpha=0.85)
        ax.plot(t, y_st[sl, h], label="STORM-Bz", alpha=0.85)
        ax.plot(t, y_ens[sl, h], "--", label=r"Ensemble ($\alpha=0.3$)", alpha=0.9)
        ax.set_ylabel(f"log flux ({title})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Test window index")
    fig.suptitle(f"Case study: {name}")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"fig_case_{name}.png", dpi=150)
    plt.close()
    print("wrote", name)

for name, center in events.items():
    center_w = int(center / max(len(te), 1) * n)
    panel(name, center_w)


# ============================================================
# P1: Missing-Feature Robustness (Ablation)
# ============================================================
print("\n" + "="*60)
print("P1: Missing-Feature Robustness (Ablation)")
print("="*60)

feature_groups = {
    "No Bz": [i for i, c in enumerate(sw_cols) if "bz" in c.lower()],
    "No Velocity/Density": [i for i, c in enumerate(sw_cols) if "vsw" in c.lower() or "density" in c.lower() or "pdyn" in c.lower()],
    "No Rolling Avgs": [i for i, c in enumerate(sw_cols) if "roll" in c.lower()]
}

ablation_results = []
for group_name, idxs in feature_groups.items():
    print(f"Testing Sensor Failure: {group_name}")
    _, y_tf_miss, _ = predict(tf, test_loader, dropout_idxs=idxs)
    _, y_st_miss, _ = predict(st, test_loader, dropout_idxs=idxs)
    
    for sys_name, pred in [("transformer", y_tf_miss), ("storm_bz", y_st_miss)]:
        ablation_results.append({
            "Sensor Failure": group_name, "system": sys_name,
            "PE_45min": pe_clim(y[:, 0], pred[:, 0]),
            "PE_6h": pe_clim(y[:, 1], pred[:, 1])
        })

df_ab = pd.DataFrame(ablation_results)
df_ab.to_csv(OUT / "tables" / "missing_feature_robustness.csv", index=False)
print(df_ab.to_markdown(index=False))

print(f"\nDONE! All Paper Extras saved to: {OUT}")

import nbformat as nbf
from pathlib import Path

REPO_DIR = Path(r"f:\Downloads\ieee_final_fixed")
NB_DIR = REPO_DIR / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

# 1. Eval Notebook
eval_nb = nbf.v4.new_notebook()
eval_code = """# ==============================================================================
# STORM-PhysNet — COMPLETE EVAL LOOP (Kaggle, NO training)
# Ensemble α*, hybrid, PE_st,6h, bootstrap CIs, bagging, tables, figure
# Requires: account ckpt zips under /kaggle/input/**
# ==============================================================================
import os, sys, json, shutil, subprocess, zipfile, traceback
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- paths --------------------
REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("/kaggle/working/STORM-PhysNet")
WORK = Path("/kaggle/working")
CKPT_ROOT = WORK / "checkpoints"
OUT = WORK / "eval_export"
TMP = WORK / "tmp_ckpt_extract"

for d in [CKPT_ROOT, OUT, TMP]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- clone repo --------------------
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR.resolve()))

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import prediction_efficiency, prediction_efficiency_pers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -------------------- ingest checkpoints from Kaggle input --------------------
def ingest_checkpoints():
    n = 0
    search_roots = [Path("/kaggle/input"), TMP]
    # extract zips first
    for zpath in Path("/kaggle/input").rglob("*.zip"):
        if "ckpt" not in zpath.name.lower() and "checkpoint" not in zpath.name.lower():
            # still try account ckpt zips by name
            if "STORM_account" not in zpath.name and "ckpt" not in zpath.name:
                continue
        try:
            with zipfile.ZipFile(zpath, "r") as zr:
                zr.extractall(TMP / zpath.stem)
        except Exception as e:
            print("zip skip", zpath, e)

    for root in search_roots:
        if not root.exists():
            continue
        for pt in root.rglob("*_best.pt"):
            parts = pt.parts
            # .../model_name/seed_XX/*.pt
            seed_dir = pt.parent.name
            model_dir = pt.parent.parent.name
            if not seed_dir.startswith("seed_"):
                continue
            dest = CKPT_ROOT / model_dir / seed_dir
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / pt.name
            if not target.exists():
                shutil.copy2(pt, target)
                n += 1
    print(f"Ingested/copied {n} new checkpoints into {CKPT_ROOT}")
    # summary
    for m in sorted(p.name for p in CKPT_ROOT.iterdir() if p.is_dir()):
        nseed = len(list((CKPT_ROOT / m).glob("seed_*")))
        print(f"  {m}: {nseed} seed dirs")

ingest_checkpoints()

# -------------------- data (same pipeline as training) --------------------
with open(REPO_DIR / "configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

goes_df = read_goes_directory(str(REPO_DIR / "datasets/goes"))
wind_df = read_wind_directory(str(REPO_DIR / "datasets/omni"))
raw_df = goes_df.join(wind_df, how="inner")
train_df, val_df, test_df = Preprocessor().fit_transform(raw_df)

seq_len = int(base_config["data"]["sequence_length"])
batch_size = int(base_config["training"].get("batch_size", 64))
storm_weight = float(base_config["training"].get("storm_weight", 12.0))

train_loader, val_loader, test_loader = make_dataloaders(
    train_df, val_df, test_df,
    seq_len=seq_len,
    batch_size=batch_size,
    storm_weight=storm_weight,
)
n_sw = int(next(iter(train_loader))["x_sw"].shape[-1])
print("n_sw", n_sw, "test batches", len(test_loader))

SEEDS = list(range(42, 57))

# model registry: name -> build kwargs + how to find ckpt
MODEL_SPECS = {
    "lstm": dict(model_type="lstm", gate_type="bz", ablation="none", match=False, spectral=False),
    "transformer": dict(model_type="transformer", gate_type="bz", ablation="none", match=False, spectral=False),
    "transformer_matched": dict(model_type="transformer", gate_type="bz", ablation="none", match=True, spectral=False),
    "storm_bz": dict(model_type="storm_physnet", gate_type="bz", ablation="none", match=False, spectral=False),
    "storm_no_delay": dict(model_type="storm_physnet", gate_type="bz", ablation="no_delay", match=False, spectral=False),
    "storm_no_physics": dict(model_type="storm_physnet", gate_type="bz", ablation="no_physics", match=False, spectral=False),
    "storm_no_gate": dict(model_type="storm_physnet", gate_type="bz", ablation="no_bz_gate", match=False, spectral=False),
    "storm_cathode": dict(model_type="storm_physnet", gate_type="cathode_anode", ablation="none", match=False, spectral=False),
    "storm_cathode_spec": dict(model_type="storm_physnet", gate_type="cathode_anode", ablation="none", match=False, spectral=True),
    "storm_radiotrophic": dict(model_type="storm_physnet", gate_type="radiotrophic", ablation="none", match=False, spectral=False),
}

def make_cfg(spec):
    cfg = deepcopy(base_config)
    cfg["model_type"] = spec["model_type"]
    cfg["ablation"] = spec["ablation"]
    cfg["match_storm_capacity"] = bool(spec["match"])
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = spec["gate_type"]
    cfg["model"]["use_spectral_head"] = bool(spec["spectral"])
    cfg["training"]["checkpoint_dir"] = "checkpoints/_eval_tmp"
    return cfg

def find_ckpt(name, seed):
    d = CKPT_ROOT / name / f"seed_{seed}"
    if not d.exists():
        return None
    pts = sorted(d.glob("*_best.pt"))
    return pts[0] if pts else None

def build_and_load(name, seed):
    spec = MODEL_SPECS[name]
    cfg = make_cfg(spec)
    trainer = Trainer(cfg)
    model = trainer.build_model(n_sw)
    pt = find_ckpt(name, seed)
    if pt is None:
        raise FileNotFoundError(f"missing ckpt {name} seed {seed}")
    state = torch.load(pt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

@torch.no_grad()
def predict_loader(model, loader):
    \"\"\"Returns y [N,3], p [N,3], y_pers [N,3], storm [N] (window storm_flag).\"\"\"
    ys, ps, pers, storms = [], [], [], []
    for batch in loader:
        x_sw = batch["x_sw"].to(device)
        x_flux = batch["x_flux"].to(device)
        y_persist = batch["y_persist"].to(device)
        try:
            out = model(x_sw, x_flux, y_persist)
        except TypeError:
            out = model(x_sw, x_flux)
        pred = out["flux_pred"] if isinstance(out, dict) else out
        ys.append(batch["y_flux"].numpy())
        ps.append(pred.detach().cpu().numpy())
        pers.append(batch["y_persist"].numpy())
        # storm_flag: [B,1] or [B]
        sf = batch["storm_flag"].numpy().reshape(-1)
        storms.append(sf)
    y = np.concatenate(ys, 0)
    p = np.concatenate(ps, 0)
    yp = np.concatenate(pers, 0)
    st = np.concatenate(storms, 0) > 0.5
    return y, p, yp, st

def pe_pack(y, p, yp, st):
    row = {
        "PE_1h": float(prediction_efficiency(y[:, 0], p[:, 0])),
        "PE_6h": float(prediction_efficiency(y[:, 1], p[:, 1])),
        "PE_12h": float(prediction_efficiency(y[:, 2], p[:, 2])),
        "PE_pers_1h": float(prediction_efficiency_pers(y[:, 0], p[:, 0], yp[:, 0])),
    }
    if st.any():
        row["PE_st_6h"] = float(prediction_efficiency(y[st, 1], p[st, 1]))
        row["n_storm"] = int(st.sum())
    else:
        row["PE_st_6h"] = float("nan")
        row["n_storm"] = 0
    return row

def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(v, size=len(v), replace=True)
        means.append(sample.mean())
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)

# -------------------- 1) per-seed predictions cache --------------------
# Keep preds in memory for ensemble/hybrid/bagging: dict[name][seed] = (y,p,yp,st)
# y is identical across models (same test set order) — store once
CACHE = defaultdict(dict)
Y_REF = None

print("\\n=== Per-seed inference ===")
seed_rows = []
for name in MODEL_SPECS:
    for seed in SEEDS:
        pt = find_ckpt(name, seed)
        if pt is None:
            print("MISSING", name, seed)
            continue
        try:
            model = build_and_load(name, seed)
            y, p, yp, st = predict_loader(model, test_loader)
            if Y_REF is None:
                Y_REF = y
            CACHE[name][seed] = (y, p, yp, st)
            metrics = pe_pack(y, p, yp, st)
            row = {"name": name, "seed": seed, **metrics}
            seed_rows.append(row)
            print(f"{name:22s} seed={seed} PE6={metrics['PE_6h']:.4f} PEst6={metrics['PE_st_6h']:.4f}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print("FAIL", name, seed, e)
            traceback.print_exc()

df_seeds = pd.DataFrame(seed_rows)
df_seeds.to_csv(OUT / "all_seed_results_full.csv", index=False)
print(df_seeds.groupby("name").size())

# -------------------- 2) seed-mean table + bootstrap CIs --------------------
print("\\n=== Means + bootstrap CIs ===")
ci_rows = []
for name, g in df_seeds.groupby("name"):
    rec = {"name": name, "n": len(g)}
    for col in ["PE_1h", "PE_6h", "PE_12h", "PE_pers_1h", "PE_st_6h"]:
        if col not in g.columns:
            continue
        mean, lo, hi = bootstrap_ci(g[col].values, seed=hash(name + col) % 2**31)
        rec[f"{col}_mean"] = mean
        rec[f"{col}_lo"] = lo
        rec[f"{col}_hi"] = hi
        rec[f"{col}_std"] = float(np.nanstd(g[col].values, ddof=1)) if len(g) > 1 else 0.0
    ci_rows.append(rec)

df_ci = pd.DataFrame(ci_rows)
df_ci.to_csv(OUT / "table_means_bootstrap_ci.csv", index=False)
print(df_ci[["name", "PE_1h_mean", "PE_6h_mean", "PE_12h_mean", "PE_st_6h_mean"]].round(3))

# -------------------- 3) true bagging --------------------
print("\\n=== True bagging ===")
bag_rows = []
for name in MODEL_SPECS:
    seeds_avail = sorted(CACHE[name].keys())
    if len(seeds_avail) < 2:
        continue
    preds = []
    y0 = yp0 = st0 = None
    for s in seeds_avail:
        y, p, yp, st = CACHE[name][s]
        preds.append(p)
        y0, yp0, st0 = y, yp, st
    P = np.mean(np.stack(preds, 0), axis=0)
    metrics = pe_pack(y0, P, yp0, st0)
    rec = {"name": name, "n_members": len(preds), **metrics}
    bag_rows.append(rec)
    (OUT / f"BAGGED_{name}_pe.json").write_text(json.dumps(rec, indent=2))
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})

pd.DataFrame(bag_rows).to_csv(OUT / "table_bagged.csv", index=False)

# -------------------- 4) α-ensemble on SAME seed (STORM-Bz + Transformer) --------------------
# Paper: sweep α on validation PE_6h, pick α*, apply once on test.
print("\\n=== α-ensemble (per seed, val-selected α) ===")
ALPHAS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

def predict_val(model):
    return predict_loader(model, val_loader)

ens_rows = []
# need val preds — compute per seed
for seed in SEEDS:
    if seed not in CACHE["storm_bz"] or seed not in CACHE["transformer"]:
        continue
    try:
        m_s = build_and_load("storm_bz", seed)
        m_t = build_and_load("transformer", seed)
        yv_s, pv_s, _, _ = predict_val(m_s)
        yv_t, pv_t, _, _ = predict_val(m_t)
        # val PE_6h for each alpha
        best_a, best_pe = None, -1e9
        for a in ALPHAS:
            mix = a * pv_s + (1 - a) * pv_t
            pe6 = prediction_efficiency(yv_s[:, 1], mix[:, 1])
            if pe6 > best_pe:
                best_pe, best_a = pe6, a
        # test mix with α*
        y, p_s, yp, st = CACHE["storm_bz"][seed]
        _, p_t, _, _ = CACHE["transformer"][seed]
        mix_t = best_a * p_s + (1.0 - best_a) * p_t
        metrics = pe_pack(y, mix_t, yp, st)
        rec = {"name": "ensemble_alpha", "seed": seed, "alpha_star": best_a, "val_PE_6h": float(best_pe), **metrics}
        ens_rows.append(rec)
        print(f"seed {seed} α*={best_a} test PE6={metrics['PE_6h']:.4f}")
        del m_s, m_t
        torch.cuda.empty_cache()
    except Exception as e:
        print("ensemble fail", seed, e)

df_ens = pd.DataFrame(ens_rows)
if len(df_ens):
    df_ens.to_csv(OUT / "ensemble_per_seed.csv", index=False)
    # report mean over seeds of test PE at each seed's α*
    # also fixed α=0.3 diagnostic (paper used validation-selected; often 0.3)
    fixed_rows = []
    for seed in SEEDS:
        if seed not in CACHE["storm_bz"] or seed not in CACHE["transformer"]:
            continue
        y, p_s, yp, st = CACHE["storm_bz"][seed]
        _, p_t, _, _ = CACHE["transformer"][seed]
        mix = 0.3 * p_s + 0.7 * p_t
        m = pe_pack(y, mix, yp, st)
        fixed_rows.append({"seed": seed, "alpha": 0.3, **m})
    df_fixed = pd.DataFrame(fixed_rows)
    df_fixed.to_csv(OUT / "ensemble_alpha0.3_per_seed.csv", index=False)
    summary_ens = {
        "alpha_star_mean_test": df_ens[["PE_1h", "PE_6h", "PE_12h", "PE_st_6h"]].mean().to_dict(),
        "alpha_star_values": df_ens["alpha_star"].value_counts().to_dict(),
        "alpha_0.3_mean_test": df_fixed[["PE_1h", "PE_6h", "PE_12h", "PE_st_6h"]].mean().to_dict(),
    }
    (OUT / "ensemble_summary.json").write_text(json.dumps(summary_ens, indent=2))
    print("ensemble summary", json.dumps(summary_ens, indent=2))

# -------------------- 5) hybrid: 1h STORM, 6h/12h Transformer --------------------
print("\\n=== Hybrid short-STORM / long-TF ===")
hyb_rows = []
for seed in SEEDS:
    if seed not in CACHE["storm_bz"] or seed not in CACHE["transformer"]:
        continue
    y, p_s, yp, st = CACHE["storm_bz"][seed]
    _, p_t, _, _ = CACHE["transformer"][seed]
    mix = p_t.copy()
    mix[:, 0] = p_s[:, 0]  # 1 h from STORM
    m = pe_pack(y, mix, yp, st)
    hyb_rows.append({"name": "hybrid_short_storm_long_tf", "seed": seed, **m})

df_hyb = pd.DataFrame(hyb_rows)
if len(df_hyb):
    df_hyb.to_csv(OUT / "hybrid_per_seed.csv", index=False)
    print(df_hyb[["PE_1h", "PE_6h", "PE_12h", "PE_st_6h"]].mean())

# -------------------- 6) paper-style mean table --------------------
ORDER = [
    "lstm", "transformer", "storm_bz", "storm_no_delay", "storm_no_physics", "storm_no_gate",
    "transformer_matched", "storm_cathode", "storm_cathode_spec", "storm_radiotrophic",
]
means = df_seeds.groupby("name")[["PE_1h", "PE_6h", "PE_12h", "PE_pers_1h", "PE_st_6h"]].mean()
means = means.reindex([n for n in ORDER if n in means.index])
means.to_csv(OUT / "table_main_means.csv")
print(means.round(3))

# -------------------- 7) figure --------------------
plot_names = [n for n in ["lstm", "transformer", "storm_bz", "transformer_matched", "storm_cathode", "storm_radiotrophic"] if n in df_seeds["name"].unique()]
if plot_names:
    m = df_seeds.groupby("name")[["PE_1h", "PE_6h", "PE_12h"]].mean().reindex(plot_names)
    s = df_seeds.groupby("name")[["PE_1h", "PE_6h", "PE_12h"]].std().reindex(plot_names)
    labels = {
        "lstm": "LSTM", "transformer": "Transformer", "storm_bz": "STORM-Bz",
        "transformer_matched": "TF matched", "storm_cathode": "RDG", "storm_radiotrophic": "SDG",
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(plot_names))
    w = 0.25
    for i, h in enumerate(["PE_1h", "PE_6h", "PE_12h"]):
        ax.bar(x + (i - 1) * w, m[h].values, w, yerr=s[h].values, capsize=3, label=h.replace("PE_", ""))
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in plot_names], rotation=15, ha="right")
    ax.set_ylabel(r"PE$_{clim}$")
    ax.set_title("Per-horizon PE (mean ± std over seeds)")
    ax.legend()
    ax.set_ylim(0.82, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_horizon_pe.png", dpi=200)
    plt.close()

# -------------------- 8) master summary --------------------
master = {
    "n_seed_rows": int(len(df_seeds)),
    "models": sorted(df_seeds["name"].unique().tolist()),
    "seeds": SEEDS,
    "means": means.round(6).to_dict() if len(means) else {},
    "note": "PE_st_6h uses batch storm_flag>0.5; ensemble α* selected per seed on val PE_6h; hybrid uses STORM 1h + TF 6h/12h",
}
(OUT / "eval_master_summary.json").write_text(json.dumps(master, indent=2))

shutil.make_archive(str(WORK / "STORM_EVAL_COMPLETE"), "zip", OUT)
print("\\nDONE — download /kaggle/working/STORM_EVAL_COMPLETE.zip")
print("Files:", sorted(p.name for p in OUT.iterdir()))
"""
eval_nb.cells.append(nbf.v4.new_code_cell(eval_code))
with open(NB_DIR / "STORM_Eval_Complete.ipynb", "w", encoding="utf-8") as f:
    nbf.write(eval_nb, f)


# 2. GRASP Transfer Notebook
grasp_nb = nbf.v4.new_notebook()
grasp_code = """# ==============================================================================
# STORM-PhysNet — GRASP transfer on NEW GOES checkpoints (Kaggle)
# Zero-shot + heads-only fine-tune | same PE math as GOES tables
# ==============================================================================
import os, sys, json, shutil, subprocess, zipfile, traceback
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("/kaggle/working/STORM-PhysNet")
WORK = Path("/kaggle/working")
CKPT_ROOT = WORK / "checkpoints"
OUT = WORK / "grasp_export"
TMP = WORK / "tmp_ckpt_extract"

# -------------------- user knobs --------------------
SEEDS = list(range(42, 57))
# Models to transfer (must exist under checkpoints/<name>/seed_XX/)
TRANSFER_MODELS = ["storm_bz", "transformer_matched"]  # drop matched if missing
GRASP_EPOCHS = 20
GRASP_LR = 1e-4
# ---------------------------------------------------

for d in [CKPT_ROOT, OUT, TMP]:
    d.mkdir(parents=True, exist_ok=True)

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR.resolve()))

from src.data.cdf_reader import (
    read_goes_directory, read_wind_directory, read_grasp_directory,
)
from src.data.preprocessor import Preprocessor
from src.data.dataloader import FluxDataset, make_dataloaders
from src.training.trainer import Trainer
from src.training.physics_loss import PhysicsInformedLoss
from src.evaluation.metrics import prediction_efficiency, prediction_efficiency_pers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# -------------------- ingest GOES checkpoints --------------------
def ingest_checkpoints():
    for zpath in Path("/kaggle/input").rglob("*.zip"):
        if "ckpt" not in zpath.name.lower() and "STORM_account" not in zpath.name:
            continue
        try:
            with zipfile.ZipFile(zpath, "r") as zr:
                zr.extractall(TMP / zpath.stem)
        except Exception as e:
            print("zip skip", zpath.name, e)

    n = 0
    for root in [Path("/kaggle/input"), TMP]:
        if not root.exists():
            continue
        for pt in root.rglob("*_best.pt"):
            seed_dir = pt.parent.name
            model_dir = pt.parent.parent.name
            if not seed_dir.startswith("seed_"):
                continue
            dest = CKPT_ROOT / model_dir / seed_dir
            dest.mkdir(parents=True, exist_ok=True)
            tgt = dest / pt.name
            if not tgt.exists():
                shutil.copy2(pt, tgt)
                n += 1
    print(f"Copied {n} checkpoints")
    for m in sorted(p.name for p in CKPT_ROOT.iterdir() if p.is_dir()):
        print(f"  {m}: {len(list((CKPT_ROOT/m).glob('seed_*')))} seeds")

ingest_checkpoints()

# -------------------- config --------------------
with open(REPO_DIR / "configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

seq_len = int(base_config["data"]["sequence_length"])
batch_size = int(base_config["training"].get("batch_size", 64))
grasp_epochs = int(base_config.get("transfer", {}).get("grasp_epochs", GRASP_EPOCHS))
grasp_lr = float(base_config.get("transfer", {}).get("grasp_lr", GRASP_LR))

# -------------------- GOES fit (scaler only on GOES) --------------------
goes_df = read_goes_directory(str(REPO_DIR / "datasets/goes"))
omni_df = read_wind_directory(str(REPO_DIR / "datasets/omni"))
goes_raw = goes_df.join(omni_df, how="inner")
pre = Preprocessor()
train_g, val_g, test_g = pre.fit_transform(goes_raw)
print("GOES splits", len(train_g), len(val_g), len(test_g))

# n_sw from GOES loaders (feature width must match checkpoints)
_, _, goes_test_loader = make_dataloaders(
    train_g, val_g, test_g, seq_len=seq_len, batch_size=batch_size,
)
n_sw = int(next(iter(goes_test_loader))["x_sw"].shape[-1])
print("n_sw", n_sw)

# -------------------- GRASP + OMNI --------------------
grasp_flux = read_grasp_directory(str(REPO_DIR / "datasets/grasp"))
if grasp_flux.empty:
    raise SystemExit("No GRASP files in datasets/grasp")

# hourly mean flux
grasp_h = grasp_flux.resample("1h").mean()
grasp_h = grasp_h.dropna(subset=["flux_gt2mev"])
# join OMNI on intersection (same solar-wind drivers as GOES pipeline)
grasp_joined = grasp_h.join(omni_df, how="inner")
grasp_joined = grasp_joined.sort_index()
print("GRASP+OMNI rows", len(grasp_joined),
      "range", grasp_joined.index.min(), "→", grasp_joined.index.max())

# transform with GOES-fitted scaler (domain shift handled by fine-tune)
grasp_scaled = pre.transform(grasp_joined.copy())

# chronological 70/15/15 on GRASP series
n = len(grasp_scaled)
i1, i2 = int(0.70 * n), int(0.85 * n)
g_train = grasp_scaled.iloc[:i1]
g_val = grasp_scaled.iloc[i1:i2]
g_test = grasp_scaled.iloc[i2:]
print("GRASP splits", len(g_train), len(g_val), len(g_test))

def make_grasp_loaders(bs=batch_size):
    train_ds = FluxDataset(g_train, seq_len=seq_len)
    val_ds = FluxDataset(g_val, seq_len=seq_len)
    test_ds = FluxDataset(g_test, seq_len=seq_len)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False)
    return train_loader, val_loader, test_loader

g_train_loader, g_val_loader, g_test_loader = make_grasp_loaders()
print("GRASP windows", len(g_train_loader.dataset), len(g_val_loader.dataset), len(g_test_loader.dataset))

# -------------------- model specs (match GOES retrain) --------------------
SPECS = {
    "storm_bz": dict(
        model_type="storm_physnet", gate_type="bz", ablation="none",
        match=False, spectral=False,
    ),
    "transformer_matched": dict(
        model_type="transformer", gate_type="bz", ablation="none",
        match=True, spectral=False,
    ),
    "transformer": dict(
        model_type="transformer", gate_type="bz", ablation="none",
        match=False, spectral=False,
    ),
}

def make_cfg(spec):
    cfg = deepcopy(base_config)
    cfg["model_type"] = spec["model_type"]
    cfg["ablation"] = spec["ablation"]
    cfg["match_storm_capacity"] = bool(spec["match"])
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = spec["gate_type"]
    cfg["model"]["use_spectral_head"] = bool(spec["spectral"])
    cfg["training"]["checkpoint_dir"] = "checkpoints/_grasp_tmp"
    cfg.setdefault("transfer", {})
    cfg["transfer"]["grasp_checkpoint_dir"] = "checkpoints/grasp_tmp"
    return cfg

def find_goes_ckpt(name, seed):
    d = CKPT_ROOT / name / f"seed_{seed}"
    if not d.exists():
        return None
    pts = sorted(d.glob("*_best.pt"))
    return pts[0] if pts else None

def build_load_goes(name, seed):
    if name not in SPECS:
        raise KeyError(name)
    cfg = make_cfg(SPECS[name])
    trainer = Trainer(cfg)
    model = trainer.build_model(n_sw)
    pt = find_goes_ckpt(name, seed)
    if pt is None:
        raise FileNotFoundError(f"No GOES ckpt for {name} seed {seed}")
    state = torch.load(pt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, cfg

@torch.no_grad()
def eval_pe(model, loader):
    model.eval()
    ys, ps, pers, storms = [], [], [], []
    for batch in loader:
        x_sw = batch["x_sw"].to(device)
        x_flux = batch["x_flux"].to(device)
        y_persist = batch["y_persist"].to(device)
        try:
            out = model(x_sw, x_flux, y_persist)
        except TypeError:
            out = model(x_sw, x_flux)
        pred = out["flux_pred"] if isinstance(out, dict) else out
        ys.append(batch["y_flux"].numpy())
        ps.append(pred.cpu().numpy())
        pers.append(batch["y_persist"].numpy())
        storms.append(batch["storm_flag"].numpy().reshape(-1))
    y = np.concatenate(ys, 0)
    p = np.concatenate(ps, 0)
    yp = np.concatenate(pers, 0)
    st = np.concatenate(storms, 0) > 0.5
    row = {
        "PE_1h": float(prediction_efficiency(y[:, 0], p[:, 0])),
        "PE_6h": float(prediction_efficiency(y[:, 1], p[:, 1])),
        "PE_12h": float(prediction_efficiency(y[:, 2], p[:, 2])),
        "PE_pers_1h": float(prediction_efficiency_pers(y[:, 0], p[:, 0], yp[:, 0])),
    }
    if st.any():
        row["PE_st_6h"] = float(prediction_efficiency(y[st, 1], p[st, 1]))
        row["n_storm"] = int(st.sum())
    else:
        row["PE_st_6h"] = float("nan")
        row["n_storm"] = 0
    return row

def fine_tune_heads(model, train_loader, val_loader, epochs, lr, save_path):
    \"\"\"Paper protocol: freeze encoder/delay/gate; train heads only.\"\"\"
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    else:
        # VanillaTransformer / LSTM: freeze all but last linear head block
        for n, p in model.named_parameters():
            p.requires_grad = False
        # unfreeze common head names
        for n, p in model.named_parameters():
            if any(k in n.lower() for k in ["head", "fc", "forecast", "out"]):
                p.requires_grad = True
        # if nothing unfrozen, train full model at low LR (baseline fallback)
        if not any(p.requires_grad for p in model.parameters()):
            for p in model.parameters():
                p.requires_grad = True
            print("[Transfer] WARNING: no head modules found; full fine-tune at low LR")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Transfer] trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.Adam(trainable, lr=lr)
    loss_fn = PhysicsInformedLoss().to(device)

    best_val = float("inf")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            try:
                out = model(batch["x_sw"], batch["x_flux"], batch["y_persist"])
            except TypeError:
                out = model(batch["x_sw"], batch["x_flux"])
            loss, _ = loss_fn(out, batch, batch["x_sw"])
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                try:
                    out = model(batch["x_sw"], batch["x_flux"], batch["y_persist"])
                except TypeError:
                    out = model(batch["x_sw"], batch["x_flux"])
                loss, _ = loss_fn(out, batch, batch["x_sw"])
                va.append(float(loss.item()))
        val_loss = float(np.mean(va)) if va else float("inf")
        print(f"  epoch {epoch:02d} train={np.mean(tr_losses):.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

    # reload best
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    if hasattr(model, "unfreeze_all"):
        model.unfreeze_all()
    else:
        for p in model.parameters():
            p.requires_grad = True
    return model

# -------------------- main loop --------------------
rows = []
for name in TRANSFER_MODELS:
    if name not in SPECS:
        print("skip unknown", name)
        continue
    n_found = sum(find_goes_ckpt(name, s) is not None for s in SEEDS)
    print(f"\\n===== {name}: {n_found}/15 GOES ckpts =====")
    if n_found == 0:
        continue

    for seed in SEEDS:
        if find_goes_ckpt(name, seed) is None:
            print("missing GOES ckpt", name, seed)
            continue
        try:
            print(f"\\n--- {name} seed={seed} ---")
            model, cfg = build_load_goes(name, seed)

            # Zero-shot on GRASP test
            zs = eval_pe(model, g_test_loader)
            print("  zero-shot", {k: round(v, 4) if isinstance(v, float) else v for k, v in zs.items()})

            # Fine-tune heads
            ft_path = OUT / "ckpts" / name / f"seed_{seed}" / "grasp_best.pt"
            model = fine_tune_heads(
                model, g_train_loader, g_val_loader,
                epochs=grasp_epochs, lr=grasp_lr, save_path=ft_path,
            )
            ft = eval_pe(model, g_test_loader)
            print("  fine-tune", {k: round(v, 4) if isinstance(v, float) else v for k, v in ft.items()})

            rec = {
                "name": name,
                "seed": seed,
                "zero_PE_1h": zs["PE_1h"],
                "zero_PE_6h": zs["PE_6h"],
                "zero_PE_12h": zs["PE_12h"],
                "zero_PE_pers_1h": zs["PE_pers_1h"],
                "zero_PE_st_6h": zs["PE_st_6h"],
                "ft_PE_1h": ft["PE_1h"],
                "ft_PE_6h": ft["PE_6h"],
                "ft_PE_12h": ft["PE_12h"],
                "ft_PE_pers_1h": ft["PE_pers_1h"],
                "ft_PE_st_6h": ft["PE_st_6h"],
                "gain_PE_6h": ft["PE_6h"] - zs["PE_6h"],
                "gain_PE_12h": ft["PE_12h"] - zs["PE_12h"],
            }
            rows.append(rec)
            seed_dir = OUT / "seeds" / name
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / f"seed_{seed}.json").write_text(json.dumps(rec, indent=2))

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print("FAIL", name, seed, e)
            traceback.print_exc()

df = pd.DataFrame(rows)
df.to_csv(OUT / "grasp_all_seeds.csv", index=False)
print("\\n=== per-seed table ===")
print(df.head())

# -------------------- summary + paired tests --------------------
summary_rows = []
for name, g in df.groupby("name"):
    rec = {"name": name, "n": len(g)}
    for col in g.columns:
        if col in ("name", "seed"):
            continue
        rec[f"{col}_mean"] = float(g[col].mean())
        rec[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
    # paired t-test zero vs ft for 6h and 12h
    try:
        from scipy import stats
        for h in ["PE_6h", "PE_12h", "PE_1h"]:
            t, p = stats.ttest_rel(g[f"ft_{h}"], g[f"zero_{h}"])
            rec[f"paired_t_{h}"] = float(t)
            rec[f"paired_p_{h}"] = float(p)
    except Exception:
        pass
    summary_rows.append(rec)
    print(name, {k: rec[k] for k in rec if "mean" in k or k == "n"})

sdf = pd.DataFrame(summary_rows)
sdf.to_csv(OUT / "grasp_summary.csv", index=False)
(OUT / "grasp_summary.json").write_text(sdf.to_json(orient="records", indent=2))

# paper-style Table II for storm_bz
if (df["name"] == "storm_bz").any():
    g = df[df["name"] == "storm_bz"]
    table = pd.DataFrame({
        "Horizon": ["1 h", "6 h", "12 h"],
        "Zero-shot": [
            f"{g['zero_PE_1h'].mean():.3f} ± {g['zero_PE_1h'].std():.3f}",
            f"{g['zero_PE_6h'].mean():.3f} ± {g['zero_PE_6h'].std():.3f}",
            f"{g['zero_PE_12h'].mean():.3f} ± {g['zero_PE_12h'].std():.3f}",
        ],
        "Fine-tuned": [
            f"{g['ft_PE_1h'].mean():.3f} ± {g['ft_PE_1h'].std():.3f}",
            f"{g['ft_PE_6h'].mean():.3f} ± {g['ft_PE_6h'].std():.3f}",
            f"{g['ft_PE_12h'].mean():.3f} ± {g['ft_PE_12h'].std():.3f}",
        ],
        "Gain": [
            f"{(g['ft_PE_1h']-g['zero_PE_1h']).mean():+.3f}",
            f"{(g['ft_PE_6h']-g['zero_PE_6h']).mean():+.3f}",
            f"{(g['ft_PE_12h']-g['zero_PE_12h']).mean():+.3f}",
        ],
    })
    table.to_csv(OUT / "table_grasp_storm_bz.csv", index=False)
    print(table)

# domain-gap figure
if (df["name"] == "storm_bz").any():
    g = df[df["name"] == "storm_bz"]
    horizons = ["1h", "6h", "12h"]
    zs = [g["zero_PE_1h"].mean(), g["zero_PE_6h"].mean(), g["zero_PE_12h"].mean()]
    ft = [g["ft_PE_1h"].mean(), g["ft_PE_6h"].mean(), g["ft_PE_12h"].mean()]
    zse = [g["zero_PE_1h"].std(), g["zero_PE_6h"].std(), g["zero_PE_12h"].std()]
    fte = [g["ft_PE_1h"].std(), g["ft_PE_6h"].std(), g["ft_PE_12h"].std()]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, zs, w, yerr=zse, capsize=4, label="Zero-shot (GOES→GRASP)")
    ax.bar(x + w/2, ft, w, yerr=fte, capsize=4, label="Fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(["1 h", "6 h", "12 h"])
    ax.set_ylabel(r"PE$_{clim}$")
    ax.set_title("GRASP transfer (15 seeds, new GOES weights)")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grasp_domain_gap.png", dpi=200)
    plt.close()

shutil.make_archive(str(WORK / "STORM_GRASP_TRANSFER"), "zip", OUT)
print("\\nDONE — download /kaggle/working/STORM_GRASP_TRANSFER.zip")
print("Files:", sorted(p.name for p in OUT.iterdir()))
"""
grasp_nb.cells.append(nbf.v4.new_code_cell(grasp_code))
with open(NB_DIR / "STORM_GRASP_Transfer.ipynb", "w", encoding="utf-8") as f:
    nbf.write(grasp_nb, f)

print("Evaluation notebooks created successfully.")

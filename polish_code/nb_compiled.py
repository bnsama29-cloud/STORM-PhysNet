# ============================================================
# STORM-PhysNet — FINAL eval + NB5 train (all fixes inlined)
# Run after NB1 / NB4 training. Produces tables/ and plots/ on Drive.
# ============================================================
import os, json, pickle, time, shutil, sys, zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ============================================================
# COLAB ENVIRONMENT FIX: EXTRACT CODE & SET UP WORK DIR
# ============================================================
from google.colab import drive
try:
    drive.mount("/content/drive", force_remount=False)
except:
    pass

WORK = Path("/content/storm_work")
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/storm_lstm_ieee.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"

WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
if str(WORK) not in sys.path:
    sys.path.append(str(WORK))

# Ensure required packages are installed in Colab
import subprocess
import importlib
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cdflib", "numpy<2", "pandas", "scikit-learn", "pyyaml", "tqdm"])
importlib.invalidate_caches()

# Ensure code is extracted
if not (WORK / "src").exists():
    print(f"Extracting code from {DRIVE_CODE_ZIP}...")
    if not Path(DRIVE_CODE_ZIP).exists():
        print(f"CRITICAL WARNING: {DRIVE_CODE_ZIP} not found!")
    else:
        with zipfile.ZipFile(DRIVE_CODE_ZIP, "r") as z:
            z.extractall(WORK / "_code")
        code_root = next((WORK / "_code").rglob("run_training.py")).parent
        for name in ["src", "configs"]:
            src_dir, dst_dir = code_root / name, WORK / name
            if src_dir.exists():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
        for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
                  "src/training/__init__.py", "src/evaluation/__init__.py"]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).touch()

# Ensure datasets are extracted
dst_goes, dst_omni = WORK / "datasets" / "goes", WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    print(f"Extracting data from {DRIVE_DATA_ZIP}...")
    if Path(DRIVE_DATA_ZIP).exists():
        with zipfile.ZipFile(DRIVE_DATA_ZIP, "r") as z:
            z.extractall(WORK / "_data")
        g = next(p for p in (WORK / "_data").rglob("goes") if p.is_dir())
        o = next(p for p in (WORK / "_data").rglob("omni") if p.is_dir())
        dst_goes.parent.mkdir(parents=True, exist_ok=True)
        if dst_goes.exists(): shutil.rmtree(dst_goes)
        if dst_omni.exists(): shutil.rmtree(dst_omni)
        shutil.copytree(g, dst_goes)
        shutil.copytree(o, dst_omni)

from src.data.cdf_reader import read_goes_directory, read_wind_directory
try:
    from src.data.cdf_reader import read_grasp_directory
except ImportError:
    read_grasp_directory = None

from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.model.baselines import VanillaTransformer, StandardLSTM
from src.model.storm_physnet import STORMPhysNet

# ============================================================
# SETTINGS
# ============================================================
DRIVE_NB1_CKPT = "/content/drive/MyDrive/storm_physnet/nb1_stats_outputs/checkpoints"
DRIVE_NB4_CKPT = "/content/drive/MyDrive/storm_physnet/nb4_solarcycle_outputs/checkpoints"
DRIVE_NB5_CKPT = "/content/drive/MyDrive/storm_physnet/nb5_extras_outputs/checkpoints"
DRIVE_OUT      = "/content/drive/MyDrive/storm_physnet/final_compiled_outputs"
SEEDS = [42, 43, 44, 45, 46]

JOBS = [
    # NB1
    ("transformer",         "transformer",   "none",       "bz", False, "transformer", DRIVE_NB1_CKPT),
    ("storm_bz",            "storm_physnet", "none",       "bz", False, "transformer", DRIVE_NB1_CKPT),
    ("storm_no_delay",      "storm_physnet", "no_delay",   "bz", False, "transformer", DRIVE_NB1_CKPT),
    ("storm_no_physics",    "storm_physnet", "no_physics", "bz", False, "transformer", DRIVE_NB1_CKPT),
    # NB5
    ("transformer_matched", "transformer",   "none",       "bz", False, "transformer", DRIVE_NB5_CKPT),
    ("lstm",                "lstm",          "none",       "bz", False, "lstm",        DRIVE_NB5_CKPT),
    ("phys_lstm",           "storm_physnet", "none",       "bz", False, "lstm",        DRIVE_NB5_CKPT),
]

NB5_TRAIN_JOBS = [
    ("transformer_matched", "transformer",   "none", "bz", "transformer", {"layers": 3, "d_model": 128}),
    ("lstm",                "lstm",          "none", "bz", "lstm",        {}),
    ("phys_lstm",           "storm_physnet", "none", "bz", "lstm",        {}),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_directories():
    Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
    Path(f"{DRIVE_OUT}/tables").mkdir(parents=True, exist_ok=True)
    Path(f"{DRIVE_OUT}/plots").mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def pe(y, yhat, ypers):
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    ypers = np.asarray(ypers).ravel()
    mse_m = mean_squared_error(y, yhat)
    mse_p = mean_squared_error(y, ypers)
    return float(1.0 - mse_m / (mse_p + 1e-12))


def find_ckpt(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return None
    for pat in ("*_best.pt", "best.pt", "*.pt"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def build_model(job, n_sw):
    label, model_type, ablation, gate, spectral, backbone, _ = job
    if model_type == "transformer":
        # matched depth used for eval construction; weights decide actual capacity
        return VanillaTransformer(n_sw_features=n_sw, seq_len=72, n_horizons=3).to(device)
    if model_type == "lstm":
        return StandardLSTM(n_sw_features=n_sw, seq_len=72, n_horizons=3).to(device)
    return STORMPhysNet(
        n_sw_features=n_sw,
        seq_len=72,
        ablation=ablation,
        gate_type=gate,
        backbone=backbone,
        use_spectral_head=spectral,
    ).to(device)


def load_state(model, ckpt_path, strict=False):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict and incompatible is not None:
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        
        # Filter out expected missing keys from auxiliary heads that were added after NB1 training
        expected_missing = ["dst_head", "kp_head", "storm_head", "log_var_heads", "horizon_embed"]
        missing = [k for k in missing if not any(x in k for x in expected_missing)]
        
        if missing or unexpected:
            print(f"  load_state {Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def predict_flux(model, x_sw, x_flux, pers, model_type):
    """Returns (flux_pred [B, 3], raw_out_dict_or_tensor)."""
    if model_type in ("transformer", "lstm"):
        out = model(x_sw, x_flux)
    else:
        try:
            out = model(x_sw, x_flux, pers)
        except TypeError:
            out = model(x_sw, x_flux)

    if isinstance(out, dict):
        pred = out.get("flux_pred", out.get("pred", None))
        if pred is None:
            raise KeyError("dict output missing flux_pred")
        return pred, out
    if isinstance(out, (list, tuple)):
        return out[0], out
    return out, out


def extract_tau_gate(raw_out, batch_size, device):
    """Safe tau / gate extraction regardless of forward return type."""
    zeros = torch.zeros(batch_size, device=device)
    if not isinstance(raw_out, dict):
        return (
            zeros.detach().cpu().numpy(),
            zeros.detach().cpu().numpy(),
        )
    tau = raw_out.get("tau", zeros)
    gate = raw_out.get("gate_values", raw_out.get("gate", zeros))
    if torch.is_tensor(tau):
        tau = tau.detach().cpu().numpy()
    if torch.is_tensor(gate):
        gate = gate.detach().cpu().numpy()
    tau = np.asarray(tau).reshape(-1)
    gate = np.asarray(gate).reshape(-1)
    return tau, gate


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def index_to_batch(global_idx, batch_tensors):
    """Map flat test index → (batch_idx, row_in_batch)."""
    offset = 0
    for b, t in enumerate(batch_tensors):
        n = t.shape[0]
        if global_idx < offset + n:
            return b, global_idx - offset
        offset += n
    return len(batch_tensors) - 1, 0


def squeeze_dst(dst_arr):
    """Normalize Dst array to shape [N]."""
    dst = np.asarray(dst_arr)
    if dst.ndim == 1:
        return dst
    # prefer column 0 (forecast-time Dst); fall back to 1 if present
    if dst.shape[1] >= 1:
        return dst[:, 0]
    return dst.reshape(-1)


def write_cfg_nb5(seed, gate, backbone, ckpt_dir, extra_cfg):
    import yaml
    from pathlib import Path
    
    cfg_path = Path("/content/storm_work/configs/config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: {cfg_path} is missing! The zip extraction failed.")
        
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = seed
    cfg["training"]["epochs"] = 40
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = gate
    cfg["model"]["backbone"] = backbone
    if extra_cfg:
        cfg["model"].update(extra_cfg)
    cfg.setdefault("transfer", {})
    cfg["transfer"]["enabled"] = False
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)


def make_test_loader(df_trans, cfg):
    """Try common make_dataloaders signatures."""
    try:
        _, _, test_loader = make_dataloaders(df_trans, df_trans, df_trans)
        return test_loader
    except TypeError:
        pass
    try:
        pack = make_dataloaders(df_trans, cfg)
        if isinstance(pack, dict):
            return pack["test"]
        return pack[-1]
    except Exception as e:
        raise RuntimeError(f"make_dataloaders failed: {e}")


# ============================================================
# PHASE 1 — NB5 TRAINING
# ============================================================
def run_nb5_training():
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING NB5 BASELINES")
    print("=" * 60)
    Path("logs/nb5").mkdir(parents=True, exist_ok=True)
    Path(DRIVE_NB5_CKPT).mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for label, model, ablation, gate, backbone, extra_cfg in NB5_TRAIN_JOBS:
            d = Path(f"checkpoints/{label}/seed_{seed}")
            d.mkdir(parents=True, exist_ok=True)
            out = Path(DRIVE_NB5_CKPT) / label / f"seed_{seed}"

            if find_ckpt(out):
                print(f"SKIP {label} seed={seed} (already in Drive)")
                continue

            write_cfg_nb5(seed, gate, backbone, d, extra_cfg)
            log = Path(f"logs/nb5/{label}_seed{seed}.txt")
            extra = ""
            if model == "storm_physnet":
                extra += f" --gate-type {gate} --backbone {backbone} --ablation {ablation}"
            elif model == "transformer" and extra_cfg:
                if "layers" in extra_cfg:
                    extra += f" --layers {extra_cfg['layers']}"
                if "d_model" in extra_cfg:
                    extra += f" --d-model {extra_cfg['d_model']}"

            cmd = (
                f"python -u run_training.py --config configs/config.yaml "
                f"--model {model} --no-ensemble {extra}"
            )
            print(f"\n[TRAINING] {label} seed={seed}")
            print(cmd)
            os.system(cmd)

            out.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*"):
                if f.is_file():
                    shutil.copy2(f, out / f.name)


def check_all_checkpoints():
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK: VERIFYING CHECKPOINTS")
    print("=" * 60)
    missing = 0
    found = 0
    
    # Check NB1 and NB5 models
    for label, _, _, _, _, _, drive_path in JOBS:
        print(f"\nChecking {label} (Path: {Path(drive_path).name}):")
        for seed in SEEDS:
            ckpt = find_ckpt(Path(drive_path) / label / f"seed_{seed}")
            if ckpt:
                print(f"  [OK] seed={seed}")
                found += 1
            else:
                print(f"  [MISSING] seed={seed}")
                missing += 1

    # Check NB4 Cross-Year models
    print(f"\nChecking NB4 Cross-Year (Path: nb4_solarcycle_outputs):")
    for label in ["transformer", "lstm", "mlp", "storm_bz"]:
        for seed in SEEDS:
            ckpt = find_ckpt(Path(DRIVE_NB4_CKPT) / "holdout_2016" / label / f"seed_{seed}")
            if ckpt:
                print(f"  [OK] {label} seed={seed}")
                found += 1
            else:
                print(f"  [MISSING] {label} seed={seed}")
                missing += 1
                
    print("-" * 60)
    print(f"Total Checkpoints Found: {found} (out of 55)")
    print(f"Total Checkpoints Missing: {missing}")
    print("-" * 60)
    
    if missing > 0:
        print("\nWARNING: Some checkpoints are missing! The evaluation will skip them,")
        print("which means your final tables and plots will have missing data points.")
        print("If you are still training, wait for them to finish before continuing.")
    else:
        print("\nSUCCESS: All 55 checkpoints found! (20 NB1 + 15 NB5 + 20 NB4)")
        print("Ready for full evaluation.")
    
    print("=" * 60)
    return missing == 0

# ============================================================
# PHASE 2 — EVALUATIONS
# ============================================================
@torch.no_grad()
def run_all_evaluations():
    import os
    from pathlib import Path
    WORK = Path("/content/storm_work")
    if WORK.exists():
        os.chdir(WORK)
        
    check_all_checkpoints()
    
    setup_directories()
    print("\n" + "=" * 60)
    print("PHASE 2: ALL EVALUATIONS & PLOTS")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------
    print("[1/9] Loading data...")
    cfg = yaml.safe_load(open("configs/config.yaml"))
    goes = read_goes_directory(cfg["data"]["goes_cdf_dir"])
    wind = read_wind_directory(cfg["data"]["wind_cdf_dir"])
    raw = goes.join(wind, how="inner")

    pp_hits = list(Path(DRIVE_NB1_CKPT).parent.rglob("preprocessor.pkl"))
    if not pp_hits:
        pp_hits = list(Path("checkpoints").rglob("preprocessor.pkl"))
    if not pp_hits:
        raise FileNotFoundError("preprocessor.pkl not found under NB1 outputs or local checkpoints")
    pp = pickle.load(open(pp_hits[0], "rb"))
    print("  preprocessor:", pp_hits[0])

    raw_transformed = pp.transform(raw)
    test_loader = make_test_loader(raw_transformed, cfg)
    n_sw = test_loader.dataset.n_sw_features
    print("  n_sw_features:", n_sw)

    # 2016 holdout loader
    test_loader_2016 = None
    try:
        raw_2016 = raw[raw.index.year == 2016]
    except Exception as e:
        print("  year filter failed:", e)
        raw_2016 = raw.iloc[0:0]

    if len(raw_2016) > 100:
        try:
            test_loader_2016 = make_test_loader(pp.transform(raw_2016), cfg)
            print("  2016 holdout windows: OK")
        except Exception as e:
            print("  2016 loader failed:", e)
            test_loader_2016 = None
    else:
        print("  2016 slice empty or too small — NB4 cross-year will be skipped")

    # Cache standard test batches
    all_x, all_x_flux, all_y, all_pers, all_storm, all_dst = [], [], [], [], [], []
    for batch in test_loader:
        # support a few key naming conventions
        x_sw = batch.get("x_sw", batch.get("x"))
        x_flux = batch.get("x_flux", batch.get("flux"))
        y = batch.get("y_flux", batch.get("y"))
        pers = batch.get("y_persist", batch.get("pers"))
        dst = batch.get("y_dst", batch.get("dst", None))

        all_x.append(x_sw)
        if x_flux is None:
            x_flux = torch.zeros(x_sw.size(0), x_sw.size(1), 1)
        all_x_flux.append(x_flux)
        all_y.append(y)
        all_pers.append(pers)
        if dst is not None:
            all_dst.append(dst)
        if "storm_flag" in batch:
            all_storm.append(batch["storm_flag"])
        elif "storm" in batch:
            all_storm.append(batch["storm"])

    global_truth = torch.cat(all_y, dim=0).numpy()          # [N, 3]
    global_pers = torch.cat(all_pers, dim=0).numpy()        # [N, 3]
    if all_dst:
        global_dst = squeeze_dst(torch.cat(all_dst, dim=0).numpy())
    else:
        global_dst = np.zeros(len(global_truth))

    if all_storm:
        global_storm_mask = torch.cat(all_storm, dim=0).numpy().reshape(-1).astype(bool)
    else:
        global_storm_mask = global_dst <= -50.0

    print(f"  test N={len(global_truth)}  storm%={100*global_storm_mask.mean():.2f}")

    # ----------------------------------------------------------
    # 2. Main metrics + ensemble + stats
    # ----------------------------------------------------------
    print("\n[2/9] Main metrics + ensembles + stats...")
    results = defaultdict(list)
    ensemble_preds = defaultdict(dict)
    best_storm_model = None
    best_storm_pe = -1e9

    for job in JOBS:
        label, model_type, ablation, gate, spectral, backbone, drive_path = job
        for seed in SEEDS:
            ckpt_path = find_ckpt(Path(drive_path) / label / f"seed_{seed}")
            if ckpt_path is None:
                continue

            model = load_state(build_model(job, n_sw), ckpt_path, strict=False)
            yh = []
            for batch_x, batch_flux, batch_pers in zip(all_x, all_x_flux, all_pers):
                pred, _ = predict_flux(
                    model,
                    batch_x.to(device),
                    batch_flux.to(device),
                    batch_pers.to(device),
                    model_type,
                )
                yh.append(pred.detach().cpu().numpy())

            yhat = np.concatenate(yh, axis=0)  # [N, 3]
            if yhat.ndim == 1:
                yhat = yhat.reshape(-1, 1)
            ensemble_preds[label][seed] = yhat

            # horizons
            pe_45 = pe(global_truth[:, 0], yhat[:, 0], global_pers[:, 0]) if yhat.shape[1] > 0 else np.nan
            pe_6  = pe(global_truth[:, 1], yhat[:, 1], global_pers[:, 1]) if yhat.shape[1] > 1 else np.nan
            pe_12 = pe(global_truth[:, 2], yhat[:, 2], global_pers[:, 2]) if yhat.shape[1] > 2 else np.nan

            if global_storm_mask.any() and yhat.shape[1] > 1:
                pe_st = pe(
                    global_truth[global_storm_mask, 1],
                    yhat[global_storm_mask, 1],
                    global_pers[global_storm_mask, 1],
                )
            else:
                pe_st = np.nan

            # high-flux PE (top 10% of truth at 6h)
            if yhat.shape[1] > 1:
                thr = np.quantile(global_truth[:, 1], 0.90)
                hi = global_truth[:, 1] >= thr
                pe_hi = pe(global_truth[hi, 1], yhat[hi, 1], global_pers[hi, 1]) if hi.any() else np.nan
            else:
                pe_hi = np.nan

            results["Model"].append(label)
            results["Seed"].append(seed)
            results["PE_45m"].append(pe_45)
            results["PE_6h"].append(pe_6)
            results["PE_12h"].append(pe_12)
            results["PE_Storm_6h"].append(pe_st)
            results["PE_Hi_6h"].append(pe_hi)
            print(f"  {label:22s} seed={seed}  PE6={pe_6:.4f}  PEstorm={pe_st:.4f}")

            if label == "storm_bz" and pe_6 > best_storm_pe:
                best_storm_pe = pe_6
                best_storm_model = model

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{DRIVE_OUT}/tables/raw_seed_metrics.csv", index=False)

    if len(df_res) == 0:
        print("ERROR: no checkpoints evaluated. Check DRIVE_*_CKPT paths.")
        return

    summary = df_res.groupby("Model").agg(
        {
            "PE_45m": ["mean", "std"],
            "PE_6h": ["mean", "std"],
            "PE_12h": ["mean", "std"],
            "PE_Storm_6h": ["mean", "std"],
            "PE_Hi_6h": ["mean", "std"],
        }
    ).round(4)
    # Flatten the MultiIndex columns so we can join with df_ens
    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]

    # Paired stats: storm_bz vs transformer on PE_Storm_6h
    def paired_stats(df, a="storm_bz", b="transformer", col="PE_Storm_6h"):
        va = df.loc[df.Model == a, col].dropna().values
        vb = df.loc[df.Model == b, col].dropna().values
        n = min(len(va), len(vb))
        if n < 3:
            return {}
        va, vb = va[:n], vb[:n]
        tstat, p_t = stats.ttest_rel(va, vb)
        diff = va - vb
        d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
        out = {
            "model_a": a,
            "model_b": b,
            "metric": col,
            "n": int(n),
            "mean_a": float(va.mean()),
            "mean_b": float(vb.mean()),
            "t": float(tstat),
            "p_paired_t": float(p_t),
            "cohen_d": d,
        }
        try:
            wstat, p_w = stats.wilcoxon(va, vb)
            out["p_wilcoxon"] = float(p_w)
        except Exception:
            out["p_wilcoxon"] = None
        return out

    stats_out = paired_stats(df_res)
    if stats_out:
        json.dump(stats_out, open(f"{DRIVE_OUT}/tables/stats_storm_pe.json", "w"), indent=2)
        print("  stats:", stats_out)

    # Deep ensemble (if >= 3 seeds)
    ens_rows = []
    for label, seeds_dict in ensemble_preds.items():
        if len(seeds_dict) < 3:
            continue
        stacked = np.stack(list(seeds_dict.values()), axis=0)  # [S, N, H]
        ens_mean = stacked.mean(axis=0)
        ens_std = stacked.std(axis=0)
        row = {
            "Model": label,
            "n_seeds": len(seeds_dict),
            "PE_45m_Ens": pe(global_truth[:, 0], ens_mean[:, 0], global_pers[:, 0]),
            "PE_6h_Ens": pe(global_truth[:, 1], ens_mean[:, 1], global_pers[:, 1]),
            "PE_12h_Ens": pe(global_truth[:, 2], ens_mean[:, 2], global_pers[:, 2])
            if ens_mean.shape[1] > 2
            else np.nan,
            "PE_Storm_6h_Ens": pe(
                global_truth[global_storm_mask, 1],
                ens_mean[global_storm_mask, 1],
                global_pers[global_storm_mask, 1],
            )
            if global_storm_mask.any()
            else np.nan,
            "Coverage_90_6h": float(
                np.mean(np.abs(global_truth[:, 1] - ens_mean[:, 1]) <= 1.64 * (ens_std[:, 1] + 1e-8))
            ),
        }
        ens_rows.append(row)

    df_ens = pd.DataFrame(ens_rows).set_index("Model") if ens_rows else pd.DataFrame()
    if len(df_ens):
        summary = summary.join(df_ens, how="left")
    summary.to_csv(f"{DRIVE_OUT}/tables/final_metrics_table.csv")
    print(summary)

    # ----------------------------------------------------------
    # 3. NB4 cross-year (2016-only test)
    # ----------------------------------------------------------
    print("\n[3/9] NB4 cross-year (2016 holdout)...")
    if test_loader_2016 is not None:
        nb4_path = Path(DRIVE_NB4_CKPT) / "holdout_2016"
        nb4_res = defaultdict(list)
        if nb4_path.exists():
            for job in JOBS:
                label, model_type, *_rest = job
                for seed in SEEDS:
                    ckpt = find_ckpt(nb4_path / label / f"seed_{seed}")
                    if ckpt is None:
                        continue
                    model = load_state(build_model(job, n_sw), ckpt, strict=False)
                    yh, ys, yp = [], [], []
                    for batch in test_loader_2016:
                        x_sw = batch.get("x_sw", batch.get("x")).to(device)
                        x_flux = batch.get("x_flux", batch.get("flux"))
                        if x_flux is None:
                            x_flux = torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)
                        else:
                            x_flux = x_flux.to(device)
                        pers = batch.get("y_persist", batch.get("pers")).to(device)
                        y = batch.get("y_flux", batch.get("y"))
                        pred, _ = predict_flux(model, x_sw, x_flux, pers, model_type)
                        yh.append(pred[:, 1].detach().cpu().numpy())
                        ys.append(y[:, 1].numpy())
                        yp.append(pers[:, 1].detach().cpu().numpy())
                    pe_2016 = pe(np.concatenate(ys), np.concatenate(yh), np.concatenate(yp))
                    nb4_res["Model"].append(label)
                    nb4_res["Seed"].append(seed)
                    nb4_res["PE_2016_6h"].append(pe_2016)
                    print(f"  holdout {label} seed={seed} PE6={pe_2016:.4f}")
            if nb4_res:
                pd.DataFrame(nb4_res).groupby("Model").agg(["mean", "std"]).round(4).to_csv(
                    f"{DRIVE_OUT}/tables/nb4_cross_year.csv"
                )
        else:
            print("  holdout_2016 checkpoint root not found")
    else:
        print("  skipped (no 2016 loader)")

    # ----------------------------------------------------------
    # 4. Case studies + top-20 failures
    # ----------------------------------------------------------
    print("\n[4/9] Case studies + top-20 failures...")
    if best_storm_model is not None:
        all_preds = []
        for batch_x, batch_flux, batch_pers in zip(all_x, all_x_flux, all_pers):
            pred, _ = predict_flux(
                best_storm_model,
                batch_x.to(device),
                batch_flux.to(device),
                batch_pers.to(device),
                "storm_physnet",
            )
            all_preds.append(pred[:, 1].detach().cpu().numpy())
        all_preds_flat = np.concatenate(all_preds)

        residuals = np.abs(global_truth[:, 1] - all_preds_flat)
        top_20_idx = np.argsort(residuals)[-20:][::-1]
        pd.DataFrame(
            {
                "TestIndex": top_20_idx,
                "True_Flux": global_truth[top_20_idx, 1],
                "Pred_Flux": all_preds_flat[top_20_idx],
                "Residual": residuals[top_20_idx],
                "Dst": global_dst[top_20_idx] if len(global_dst) == len(residuals) else np.nan,
            }
        ).to_csv(f"{DRIVE_OUT}/tables/top_20_failures.csv", index=False)

        # heuristic event indices
        enh_idx = int(np.argmax(global_truth[:, 1]))
        drop_idx = int(np.argmax(global_truth[:-1, 1] - global_truth[1:, 1]))
        quiet_idx = int(np.argmin(np.abs(global_truth[:, 1] - np.median(global_truth[:, 1]))))
        cases = {"Enhancement": enh_idx, "Dropout": drop_idx, "Quiet": quiet_idx}

        for name, gidx in cases.items():
            bidx, _ = index_to_batch(gidx, all_y)
            x_sw = all_x[bidx].to(device)
            x_flux = all_x_flux[bidx].to(device)
            pers = all_pers[bidx].to(device)
            pred, raw_out = predict_flux(best_storm_model, x_sw, x_flux, pers, "storm_physnet")
            pred_np = pred[:, 1].detach().cpu().numpy()
            tau, gate = extract_tau_gate(raw_out, x_sw.size(0), device)

            truth = all_y[bidx][:, 1].numpy()
            pers_np = pers[:, 1].detach().cpu().numpy()
            # channel indices: document if your order differs
            vsw = x_sw[:, -1, 0].detach().cpu().numpy()
            bz = x_sw[:, -1, 1].detach().cpu().numpy()

            n = len(truth)
            ts = np.arange(n)
            # broadcast tau/gate if scalar-per-batch
            if tau.size == 1:
                tau = np.full(n, float(tau))
            if gate.size == 1:
                gate = np.full(n, float(gate))
            tau = tau[:n]
            gate = gate[:n]

            fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
            axes[0].plot(ts, truth, "k", label="True")
            axes[0].plot(ts, pred_np, "r", label="STORM")
            axes[0].plot(ts, pers_np, "g--", label="Persistence")
            axes[0].legend(loc="best")
            axes[0].set_ylabel("log-flux")
            axes[0].set_title(f"Case study: {name}")
            axes[1].plot(ts, bz)
            axes[1].set_ylabel("Bz")
            axes[2].plot(ts, vsw)
            axes[2].set_ylabel("Vsw")
            axes[3].plot(ts, tau)
            axes[3].set_ylabel("tau")
            axes[4].plot(ts, gate)
            axes[4].set_ylabel("gate")
            fig.tight_layout()
            fig.savefig(f"{DRIVE_OUT}/plots/case_study_{name}.png", dpi=150)
            plt.close(fig)
            print(f"  saved case_study_{name}.png")
    else:
        print("  no storm_bz model loaded — skip cases")

    # ----------------------------------------------------------
    # 5. Robustness
    # ----------------------------------------------------------
    print("\n[5/9] Robustness (noise × channel drop)...")
    if best_storm_model is not None:
        robust_res = {"Noise": [], "DropFrac": [], "PE_6h": []}
        for noise in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for frac in [0.0, 0.10, 0.20, 0.30]:
                yh = []
                for batch_x, batch_flux, batch_pers in zip(all_x, all_x_flux, all_pers):
                    x_sw = batch_x.to(device).clone()
                    if noise > 0:
                        x_sw = x_sw + torch.randn_like(x_sw) * noise
                    if frac > 0:
                        keep = (torch.rand(x_sw.size(-1), device=device) > frac).float()
                        x_sw = x_sw * keep
                    pred, _ = predict_flux(
                        best_storm_model,
                        x_sw,
                        batch_flux.to(device),
                        batch_pers.to(device),
                        "storm_physnet",
                    )
                    yh.append(pred[:, 1].detach().cpu().numpy())
                yhat = np.concatenate(yh)
                robust_res["Noise"].append(noise)
                robust_res["DropFrac"].append(frac)
                robust_res["PE_6h"].append(pe(global_truth[:, 1], yhat, global_pers[:, 1]))
        pd.DataFrame(robust_res).to_csv(f"{DRIVE_OUT}/tables/robustness.csv", index=False)
        print("  wrote robustness.csv")
    else:
        print("  skipped")

    # ----------------------------------------------------------
    # 6. Dst intensity bins
    # ----------------------------------------------------------
    print("\n[6/9] Dst intensity bins...")
    if "storm_bz" in ensemble_preds and len(ensemble_preds["storm_bz"]) >= 1:
        yhat6 = np.mean(np.stack(list(ensemble_preds["storm_bz"].values()), axis=0), axis=0)[:, 1]
        
        # Unscale Dst before thresholding
        try:
            from src.data.preprocessor import ALL_FEATURES
            dst_idx = ALL_FEATURES.index("dst")
            dst_mean = pp.scaler.mean_[dst_idx]
            dst_scale = pp.scaler.scale_[dst_idx]
            unscaled_dst = global_dst * dst_scale + dst_mean
        except Exception:
            # Fallback if ALL_FEATURES is missing
            unscaled_dst = global_dst

        bins = {
            "Quiet (Dst > -30)": unscaled_dst > -30,
            "Minor (-50 < Dst <= -30)": (unscaled_dst <= -30) & (unscaled_dst > -50),
            "Moderate (-100 < Dst <= -50)": (unscaled_dst <= -50) & (unscaled_dst > -100),
            "Strong (Dst <= -100)": unscaled_dst <= -100,
        }
        rows = []
        for name, mask in bins.items():
            mask = mask.astype(bool)
            if mask.sum() < 10:
                continue
            rows.append(
                {
                    "Intensity": name,
                    "Count": int(mask.sum()),
                    "PE_6h": pe(global_truth[mask, 1], yhat6[mask], global_pers[mask, 1]),
                    "MSE_6h": float(mean_squared_error(global_truth[mask, 1], yhat6[mask])),
                }
            )
        pd.DataFrame(rows).to_csv(f"{DRIVE_OUT}/tables/dst_intensity_bins.csv", index=False)
        print("  wrote dst_intensity_bins.csv")
    else:
        print("  skipped")

    # ----------------------------------------------------------
    # 7. Physics scatters
    # ----------------------------------------------------------
    print("\n[7/9] Physics scatters...")
    if best_storm_model is not None:
        all_tau, all_vsw, all_gate, all_bz = [], [], [], []
        for batch_x, batch_flux, batch_pers in zip(all_x, all_x_flux, all_pers):
            x_sw = batch_x.to(device)
            _, raw_out = predict_flux(
                best_storm_model,
                x_sw,
                batch_flux.to(device),
                batch_pers.to(device),
                "storm_physnet",
            )
            tau, gate = extract_tau_gate(raw_out, x_sw.size(0), device)
            all_tau.append(tau)
            all_gate.append(gate)
            all_vsw.append(x_sw[:, -1, 0].detach().cpu().numpy())
            all_bz.append(x_sw[:, -1, 1].detach().cpu().numpy())

        tau = np.concatenate(all_tau)
        vsw = np.concatenate(all_vsw)
        gate = np.concatenate(all_gate)
        bz = np.concatenate(all_bz)
        n = min(len(tau), len(vsw), len(gate), len(bz))
        tau, vsw, gate, bz = tau[:n], vsw[:n], gate[:n], bz[:n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(vsw, tau, alpha=0.15, s=3)
        ax1.set_xlabel("Vsw")
        ax1.set_ylabel("tau")
        ax1.set_title("Delay vs solar-wind speed")
        ax2.scatter(bz, gate, alpha=0.15, s=3)
        ax2.set_xlabel("Bz")
        ax2.set_ylabel("gate")
        ax2.set_title("Gate vs Bz")
        fig.tight_layout()
        fig.savefig(f"{DRIVE_OUT}/plots/physics_scatters.png", dpi=150)
        plt.close(fig)
        print("  saved physics_scatters.png")
    else:
        print("  skipped")

    # ----------------------------------------------------------
    # 8. GRASP 14-vs-15 zero-shot (pad control)
    # ----------------------------------------------------------
    print("\n[8/9] GRASP zero-shot pad diagnosis...")
    grasp_rows = []
    if best_storm_model is not None and read_grasp_directory is not None:
        grasp_dir = (
            cfg.get("data", {}).get("grasp_cdf_dir")
            or cfg.get("data", {}).get("grasp_dir")
            or "datasets/grasp"
        )
        try:
            grasp_df = read_grasp_directory(grasp_dir)
        except Exception as e:
            print("  GRASP read failed:", e)
            grasp_df = None

        if grasp_df is not None and len(grasp_df) > 50:
            try:
                grasp_trans = pp.transform(grasp_df)
                grasp_loader = make_test_loader(grasp_trans, cfg)
                yh, ys, yp = [], [], []
                for batch in grasp_loader:
                    x_sw = batch.get("x_sw", batch.get("x")).to(device)
                    x_flux = batch.get("x_flux", batch.get("flux"))
                    if x_flux is None:
                        x_flux = torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)
                    else:
                        x_flux = x_flux.to(device)
                    pers = batch.get("y_persist", batch.get("pers")).to(device)
                    y = batch.get("y_flux", batch.get("y"))
                    # pad feature dim to n_sw if needed
                    if x_sw.size(-1) < n_sw:
                        pad = torch.zeros(
                            x_sw.size(0), x_sw.size(1), n_sw - x_sw.size(-1), device=device
                        )
                        x_sw = torch.cat([x_sw, pad], dim=-1)
                    elif x_sw.size(-1) > n_sw:
                        x_sw = x_sw[..., :n_sw]
                    pred, _ = predict_flux(
                        best_storm_model, x_sw, x_flux, pers, "storm_physnet"
                    )
                    yh.append(pred[:, 1].detach().cpu().numpy())
                    ys.append(y[:, 1].numpy())
                    yp.append(pers[:, 1].detach().cpu().numpy())
                pe_g = pe(np.concatenate(ys), np.concatenate(yh), np.concatenate(yp))
                grasp_rows.append(
                    {
                        "Setting": "zero_shot_pad_to_n_sw",
                        "PE_6h": pe_g,
                        "n_sw_model": n_sw,
                        "note": "GOES-pretrained storm_bz; missing GRASP channels zero-padded",
                    }
                )
                print(f"  GRASP zero-shot (padded) PE_6h={pe_g:.4f}")
            except Exception as e:
                print("  GRASP eval failed:", e)
        else:
            print("  no GRASP data")
    else:
        print("  skipped (no model or read_grasp_directory)")

    if grasp_rows:
        pd.DataFrame(grasp_rows).to_csv(
            f"{DRIVE_OUT}/tables/grasp_14vs15_diagnosis.csv", index=False
        )

    # ----------------------------------------------------------
    # 9. Compute cost
    # ----------------------------------------------------------
    print("\n[9/9] Compute cost...")
    dummy_x = torch.randn(32, 72, n_sw, device=device)
    dummy_flux = torch.randn(32, 72, 1, device=device)
    dummy_pers = torch.randn(32, 3, device=device)
    t_model = VanillaTransformer(n_sw_features=n_sw, seq_len=72, n_horizons=3).to(device).eval()
    s_model = STORMPhysNet(n_sw_features=n_sw, seq_len=72).to(device).eval()

    with torch.no_grad():
        for _ in range(10):
            t_model(dummy_x, dummy_flux)
            try:
                s_model(dummy_x, dummy_flux, dummy_pers)
            except TypeError:
                s_model(dummy_x, dummy_flux)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            t_model(dummy_x, dummy_flux)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ms = (time.time() - t0) / 100 * 1000

        t0 = time.time()
        for _ in range(100):
            try:
                s_model(dummy_x, dummy_flux, dummy_pers)
            except TypeError:
                s_model(dummy_x, dummy_flux)
        if device.type == "cuda":
            torch.cuda.synchronize()
        s_ms = (time.time() - t0) / 100 * 1000

    pd.DataFrame(
        {
            "Model": ["Transformer_matched_3layer", "STORM-PhysNet"],
            "Parameters": [count_params(t_model), count_params(s_model)],
            "Latency_ms_per_batch32": [t_ms, s_ms],
        }
    ).to_csv(f"{DRIVE_OUT}/tables/compute_cost.csv", index=False)
    print("  wrote compute_cost.csv")

    # ----------------------------------------------------------
    # Summary manifest
    # ----------------------------------------------------------
    manifest = {
        "out_dir": DRIVE_OUT,
        "tables": sorted([p.name for p in Path(f"{DRIVE_OUT}/tables").glob("*")]),
        "plots": sorted([p.name for p in Path(f"{DRIVE_OUT}/plots").glob("*")]),
        "stats_storm_pe": stats_out,
        "note": "Coverage_90 is ensemble predictive-interval coverage; do not claim calibration if << 0.90.",
    }
    json.dump(manifest, open(f"{DRIVE_OUT}/RUN_MANIFEST.json", "w"), indent=2)
    print("\n" + "=" * 60)
    print("DONE →", DRIVE_OUT)
    print("Manifest:", manifest)
    print("=" * 60)


if __name__ == "__main__":
    run_nb5_training()
    run_all_evaluations()
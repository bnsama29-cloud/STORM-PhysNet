import nbformat as nbf
from pathlib import Path

REPO_DIR = Path(r"f:\Downloads\ieee_final_fixed")
NB_DIR = REPO_DIR / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

# 1. Multi-seed training notebook
train_nb = nbf.v4.new_notebook()
train_code = """# ==============================================================================
# STORM-PhysNet — FULL RETRAIN (Kaggle, multi-account)
# Correct gate_type, spectral head, architecture-matched TF, eval JSON
# ==============================================================================
import os, sys, json, shutil, subprocess, traceback
from pathlib import Path
from copy import deepcopy

# ===================== EDIT PER ACCOUNT =====================
ACCOUNT_ID = 0
MY_SEEDS = [42, 43, 44, 45, 46]
# Account 0 uses [42, 43, 44, 45, 46]
# Account 1 uses [47, 48, 49, 50, 51]
# Account 2 uses [52, 53, 54, 55, 56]

FINAL_AGGREGATE = False
# ============================================================

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("/kaggle/working/STORM-PhysNet")

if not FINAL_AGGREGATE:
    if not REPO_DIR.exists():
        print(f"Cloning into {REPO_DIR}...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(["git", "reset", "--hard"], cwd=REPO_DIR, check=True)
        subprocess.run(["git", "pull"], cwd=REPO_DIR, check=True)

    shutil.rmtree(REPO_DIR / "checkpoints", ignore_errors=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml", "matplotlib"], check=True)

    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR.resolve()))

    import torch, yaml
    from src.data.cdf_reader import read_goes_directory, read_wind_directory
    from src.data.preprocessor import Preprocessor
    from src.data.dataloader import make_dataloaders
    from src.training.trainer import Trainer
    from src.evaluation.metrics import prediction_efficiency

    with open("configs/config.yaml") as f: base_config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    goes_df = read_goes_directory("datasets/goes")
    wind_df = read_wind_directory("datasets/omni")
    raw_df = goes_df.join(wind_df, how="inner")
    train_df, val_df, test_df = Preprocessor().fit_transform(raw_df)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        seq_len=int(base_config["data"]["sequence_length"]),
        batch_size=int(base_config["training"].get("batch_size", 64)),
    )
    n_sw = int(next(iter(train_loader))["x_sw"].shape[-1])

    def run_one(seed, name, gate_type, use_spectral, model_type="storm_physnet", ablation="none", match_capacity=False):
        cfg = deepcopy(base_config)
        cfg["model_type"] = model_type
        cfg["match_storm_capacity"] = match_capacity
        cfg["ablation"] = ablation
        cfg.setdefault("model", {})
        if gate_type in (None, "none"): cfg["ablation"] = "no_bz_gate"; cfg["model"]["gate_type"] = "bz"
        else: cfg["model"]["gate_type"] = gate_type
        cfg["model"]["use_spectral_head"] = use_spectral
        cfg["training"]["seed"] = seed
        
        results_dir = Path("results/alt_gates") if "cathode" in name or "radiotrophic" in name else Path("results/seeds")
        res_file = results_dir / name / f"seed_{seed}.json"
        ckpt_dir = Path(f"checkpoints/{name}/seed_{seed}"); ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_pt = ckpt_dir / f"{cfg['model_type']}_{cfg['model']['gate_type']}_best.pt"

        if res_file.exists() and best_pt.exists(): return

        print(f"\\n{'='*50}\\nTraining {name}  seed={seed}\\n{'='*50}")
        trainer = Trainer(cfg)
        model = trainer.build_model(n_sw).to(device)
        try: val_loss = trainer.train(model, train_loader, val_loader, device=device)
        except Exception as e: print(f"Failed: {e}"); traceback.print_exc(); return
            
        for pt in Path("checkpoints").glob("*_best.pt"): shutil.move(str(pt), best_pt)

        model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in test_loader:
                try: out = model(batch["x_sw"].to(device), batch["x_flux"].to(device), batch["y_persist"].to(device))
                except TypeError: out = model(batch["x_sw"].to(device), batch["x_flux"].to(device))
                pred = out["flux_pred"] if isinstance(out, dict) else out
                preds.append(pred.cpu().numpy()); ys.append(batch["y_flux"].numpy())
                
        P = np.concatenate(preds, 0); Y = np.concatenate(ys, 0)
        metrics = {
            "name": name, "seed": seed, "val_loss": float(val_loss),
            "PE_1h": float(prediction_efficiency(Y[:, 0], P[:, 0])),
            "PE_6h": float(prediction_efficiency(Y[:, 1], P[:, 1])),
            "PE_12h": float(prediction_efficiency(Y[:, 2], P[:, 2])),
            "n_params": sum(p.numel() for p in model.parameters())
        }
        res_file.parent.mkdir(parents=True, exist_ok=True)
        res_file.write_text(json.dumps(metrics, indent=2))

    ALL_MODELS = [
        ("lstm", "none", False, "lstm", "none", False),
        ("transformer", "none", False, "transformer", "none", False),
        ("storm_bz", "bz", False, "storm_physnet", "none", False),
        ("storm_no_delay", "bz", False, "storm_physnet", "no_delay", False),
        ("storm_no_physics", "bz", False, "storm_physnet", "no_physics", False),
        ("storm_no_gate", "none", False, "storm_physnet", "none", False),
        ("transformer_matched", "none", False, "transformer", "none", True),
        ("storm_cathode", "cathode_anode", False, "storm_physnet", "none", False),
        ("storm_cathode_spec", "cathode_anode", True, "storm_physnet", "none", False),
        ("storm_radiotrophic", "radiotrophic", False, "storm_physnet", "none", False),
    ]

    for name, g_type, spec, m_type, ab, mtch in ALL_MODELS:
        for s in MY_SEEDS: run_one(s, name, g_type, spec, m_type, ab, mtch)

    shutil.make_archive(f"/kaggle/working/STORM_account_{ACCOUNT_ID}_results", "zip", "results")
    shutil.make_archive(f"/kaggle/working/STORM_account_{ACCOUNT_ID}_ckpt", "zip", "checkpoints")
    print("DONE! Download your 2 ZIP files from the right sidebar.")
else:
    print("FINAL_AGGREGATE=True — skip training; run Cell 2 for tables/figures.")
"""
train_nb.cells.append(nbf.v4.new_code_cell(train_code))
with open(NB_DIR / "STORM_PhysNet_Colab.ipynb", "w", encoding='utf-8') as f: nbf.write(train_nb, f)

# 2. Aggregation notebook
agg_nb = nbf.v4.new_notebook()
agg_code = """# ==============================================================================
# KAGGLE AGGREGATION & BAG EVALUATION SCRIPT
# ==============================================================================
import os, sys, json, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("/kaggle/working/STORM-PhysNet")
if not REPO_DIR.exists(): subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml", "matplotlib"], check=True)

RESULTS_DIR = REPO_DIR / "results"; CKPT_DIR = REPO_DIR / "checkpoints"; OUT_DIR = REPO_DIR / "paper_export"
TEMP_DIR = Path("/kaggle/working/temp_extract")
for d in [RESULTS_DIR, CKPT_DIR, TEMP_DIR]: shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

for z in Path("/kaggle/input").rglob("*.zip"):
    try:
        with zipfile.ZipFile(z, "r") as zr: zr.extractall(TEMP_DIR)
    except: pass

found_files = 0
for search_path in [Path("/kaggle/input"), TEMP_DIR]:
    for f in search_path.rglob("*_best.pt"):
        dest = CKPT_DIR / f.parent.parent.name / f.parent.name; dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest / f.name); found_files += 1
    for f in search_path.rglob("seed_*.json"):
        dest = RESULTS_DIR / f.parent.name; dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest / f.name); found_files += 1

os.chdir(REPO_DIR); sys.path.insert(0, str(REPO_DIR.resolve()))

def load_all_json():
    rows = []
    for p in list(RESULTS_DIR.rglob("seed_*.json")):
        try: rows.append(json.loads(p.read_text()))
        except: pass
    return pd.DataFrame(rows)

df = load_all_json()
if not df.empty:
    df.to_csv(OUT_DIR / "all_seed_results.csv", index=False)
    PE_COLS = [c for c in ["PE_1h", "PE_6h", "PE_12h", "PE_pers_1h"] if c in df.columns]
    ORDER = ["lstm", "transformer", "storm_bz", "storm_no_delay", "storm_no_physics", "storm_no_gate", "transformer_matched", "storm_cathode", "storm_cathode_spec", "storm_radiotrophic"]
    df.groupby("name")[PE_COLS].agg(["mean", "std", "count"]).to_csv(OUT_DIR / "table_main_stats.csv")
    df.groupby("name")[PE_COLS].mean().reindex([n for n in ORDER if n in df["name"].unique()]).to_csv(OUT_DIR / "table_main_means.csv")
    if "n_params" in df.columns: df.groupby("name")["n_params"].first().to_csv(OUT_DIR / "table_parameter_counts.csv")

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import prediction_efficiency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("configs/config.yaml") as f: base_config = yaml.safe_load(f)
train_df, val_df, test_df = Preprocessor().fit_transform(read_goes_directory("datasets/goes").join(read_wind_directory("datasets/omni"), how="inner"))
train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, seq_len=int(base_config["data"]["sequence_length"]), batch_size=int(base_config["training"].get("batch_size", 64)))
n_sw = int(next(iter(train_loader))["x_sw"].shape[-1])

def run_bagging(name, seeds):
    try:
        cfg = deepcopy(base_config)
        cfg["model_type"] = "transformer" if "transformer" in name else ("lstm" if "lstm" in name else "storm_physnet")
        cfg["match_storm_capacity"] = "matched" in name; cfg.setdefault("model", {})
        cfg["model"]["gate_type"] = "cathode_anode" if "cathode" in name else ("radiotrophic" if "radiotrophic" in name else "bz")
        cfg["model"]["use_spectral_head"] = True if "cathode_spec" in name else False
        cfg["ablation"] = "no_bz_gate" if "no_gate" in name else ("no_delay" if "no_delay" in name else ("no_physics" if "no_physics" in name else "none"))
            
        trainer = Trainer(cfg); preds, ys = [], None
        for seed in seeds:
            pts = list(Path(f"checkpoints/{name}/seed_{seed}").glob("*_best.pt"))
            if not pts: continue
            model = trainer.build_model(n_sw); model.load_state_dict(torch.load(pts[0], map_location=device, weights_only=True)); model.to(device).eval()
            ps, ylist = [], []
            with torch.no_grad():
                for batch in test_loader:
                    try: out = model(batch["x_sw"].to(device), batch["x_flux"].to(device), batch["y_persist"].to(device))
                    except TypeError: out = model(batch["x_sw"].to(device), batch["x_flux"].to(device))
                    pred = out["flux_pred"] if isinstance(out, dict) else out
                    ps.append(pred.cpu().numpy()); ylist.append(batch["y_flux"].numpy())
            preds.append(np.concatenate(ps, 0)); ys = np.concatenate(ylist, 0)
        
        if not preds: return
        P = np.mean(np.stack(preds, 0), axis=0)
        bag = {"PE_1h": float(prediction_efficiency(ys[:, 0], P[:, 0])), "PE_6h": float(prediction_efficiency(ys[:, 1], P[:, 1])), "PE_12h": float(prediction_efficiency(ys[:, 2], P[:, 2])), "n_members": len(preds)}
        (OUT_DIR / f"BAGGED_{name}_pe.json").write_text(json.dumps(bag, indent=2))
    except Exception as e: print(f"FAILED to bag {name}: {e}")

for model_name in ["lstm", "transformer", "storm_bz", "storm_no_delay", "storm_no_physics", "storm_no_gate", "transformer_matched", "storm_cathode", "storm_cathode_spec", "storm_radiotrophic"]: run_bagging(model_name, list(range(42, 57)))

shutil.make_archive("/kaggle/working/STORM_FINAL_PAPER_DATA", "zip", OUT_DIR)
from IPython.display import display, FileLink
display(FileLink(r'STORM_FINAL_PAPER_DATA.zip'))
"""
agg_nb.cells.append(nbf.v4.new_code_cell(agg_code))
with open(NB_DIR / "kaggle_aggregate.ipynb", "w", encoding='utf-8') as f: nbf.write(agg_nb, f)

print("Notebooks successfully created/updated!")

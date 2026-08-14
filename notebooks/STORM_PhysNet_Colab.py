# !nvidia-smi

import os
import sys
from pathlib import Path

# --- Always land in the repo root and make `src` importable ---
REPO_NAME = "STORM-PhysNet"
REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"

# Colab default home
if Path("/content").exists():
    base = Path("/content")
else:
    base = Path.cwd()

repo_dir = base / REPO_NAME

# Clone once if missing
if not (repo_dir / "src").is_dir():
    # %cd {base}
    # !git clone {REPO_URL}

# %cd {repo_dir}

# Put repo root on PYTHONPATH (fixes: No module named 'src')
repo_dir = Path.cwd().resolve()
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

print("cwd:", os.getcwd())
print("src exists:", (repo_dir / "src").is_dir())
print("sys.path[0]:", sys.path[0])

# Colab already has torch/numpy/pandas; just install missing cdflib
import sys
import subprocess
import site
import importlib

print("Installing cdflib directly into the active Python environment...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "cdflib"])
importlib.reload(site)
print("cdflib successfully installed and paths refreshed!")
# optional if CDF fails:





import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import prediction_efficiency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BASE_SEED = 42
SEEDS = list(range(42, 57))  # fifteen seeds used in the papers
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

DEMO_MODE = True          # set False only for full paper-scale training
DEMO_EPOCHS = 5
FULL_EPOCHS = 40

print(f"DEMO_MODE = {DEMO_MODE}")
print("Transformer baseline is capacity-matched to STORM (see paper Methods).")



from pathlib import Path
import yaml

# Must be inside the repo root (after the clone + sys.path cell)
config_path = Path("configs/config.yaml")
assert config_path.exists(), f"Missing {config_path.resolve()} — run the setup/clone cell first"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Normalize training paths for Colab / local
ckpt = str(config["training"].get("checkpoint_dir", "checkpoints"))
log_dir = str(config["training"].get("log_dir", "logs"))

if ckpt.startswith("/kaggle") or "\\" in ckpt:
    config["training"]["checkpoint_dir"] = "checkpoints"
if log_dir.startswith("/kaggle") or "\\" in log_dir:
    config["training"]["log_dir"] = "logs"

Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
Path(config["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)

print("checkpoint_dir:", config["training"]["checkpoint_dir"])
print("log_dir:", config["training"]["log_dir"])
print("horizons:", config["data"]["forecast_horizons"])

import os, sys
from pathlib import Path

# Be in repo root
if not Path("src").is_dir():
    # %cd /content/STORM-PhysNet

repo = Path.cwd().resolve()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

# !pip install -q cdflib pyyaml

print("cwd:", repo)
print("goes:", list(Path("datasets/goes").glob("*"))[:3])
print("omni:", list(Path("datasets/omni").glob("*"))[:5])


# --- Failsafe if you skipped Cell 6 (Config) ---
if 'config' not in globals():
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
# -----------------------------------------------

print("Loading GOES and OMNI from datasets/ ...")

DATA_READY = False
n_sw_features = None
train_loader = val_loader = test_loader = None

try:
    goes_dir = config["data"].get("goes_cdf_dir", "datasets/goes")
    wind_dir = config["data"].get("wind_cdf_dir", "datasets/omni")
    print("goes_dir:", Path(goes_dir).resolve(), "exists:", Path(goes_dir).is_dir())
    print("wind_dir:", Path(wind_dir).resolve(), "exists:", Path(wind_dir).is_dir())

    goes_df = read_goes_directory(goes_dir)
    print("GOES shape:", goes_df.shape, "cols:", list(goes_df.columns)[:8])

    wind_df = read_wind_directory(wind_dir)
    print("OMNI/WIND shape:", wind_df.shape, "cols:", list(wind_df.columns)[:8])

    raw_df = goes_df.join(wind_df, how="inner")
    print("Joined shape:", raw_df.shape)
    if len(raw_df) < 1000:
        raise RuntimeError(f"Joined dataframe too small: {raw_df.shape}")

    preprocessor = Preprocessor(year_split=config["data"].get("year_split", None))
    train_df, val_df, test_df = preprocessor.fit_transform(raw_df)
    print("train/val/test:", len(train_df), len(val_df), len(test_df))

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        seq_len=config["data"]["sequence_length"],
        batch_size=config["training"].get("batch_size", config["data"].get("batch_size", 64)),
    )
    batch0 = next(iter(train_loader))
    n_sw_features = int(batch0["x_sw"].shape[-1])
    print(f"Dataloaders OK | n_sw_features={n_sw_features}")
    DATA_READY = True

except Exception as e:
    import traceback
    print("Data loading failed:")
    traceback.print_exc()
    print("Training will be skipped; paper tables still load from results/*.csv")
    DATA_READY = False



# --- Failsafe if you skipped Cell 4 ---
BASE_SEED = 42
DEMO_MODE = True
# --------------------------------------

def run_training(name, model_type="storm_physnet", gate_type="bz",
                 no_delay=False, no_physics=False, seed=BASE_SEED):
    print("=" * 70)
    print(f"Training: {name} | seed={seed} | type={model_type}")
    print("=" * 70)

    if not DATA_READY:
        print("Data not available - skipping training.")
        return None

    cfg = yaml.safe_load(yaml.dump(config))
    cfg["model_type"] = model_type
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = gate_type

    if no_delay:
        cfg["ablation"] = "no_delay"
    elif no_physics:
        cfg["ablation"] = "no_physics"
    elif gate_type in (None, "none"):
        cfg["ablation"] = "no_bz_gate"
        cfg["model"]["gate_type"] = "bz"
    else:
        cfg["ablation"] = "none"

    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = DEMO_EPOCHS if DEMO_MODE else FULL_EPOCHS
    cfg["training"]["checkpoint_dir"] = f"checkpoints/{name}/seed_{seed}"
    cfg["training"]["log_dir"] = f"logs/{name}"
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = Trainer(cfg)
    try:
        model = trainer.fit(
            train_loader,
            val_loader,
            n_sw_features=n_sw_features,
            use_ensemble=False,
        )
        print(f"Finished training {name}")
        return model
    except Exception as e:
        print(f"Training failed for {name}: {e}")
        return None




# Optional single-seed demo trains. For paper tables, prefer results/*.csv.
run_training("transformer", model_type="transformer")
run_training("lstm", model_type="lstm")
run_training("storm_bz", model_type="storm_physnet", gate_type="bz")



run_training("storm_no_delay", model_type="storm_physnet", gate_type="bz", no_delay=True)
run_training("storm_no_gate", model_type="storm_physnet", gate_type="none")
run_training("storm_no_physics", model_type="storm_physnet", gate_type="bz", no_physics=True)



def run_wider_delay_experiment(seed=42, upper_bound=2.0):
    print(f"WIDER DELAY | seed={seed} | upper_bound={upper_bound}h")
    if not DATA_READY:
        print("Data not available - skip.")
        return
    cfg = yaml.safe_load(yaml.dump(config))
    cfg["model_type"] = "storm_physnet"
    cfg.setdefault("model", {})
    cfg["model"]["delay"] = {"enabled": True}
    cfg["model"]["delay_min"] = 0.5
    cfg["model"]["delay_max"] = float(upper_bound)
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = DEMO_EPOCHS if DEMO_MODE else FULL_EPOCHS
    cfg["training"]["checkpoint_dir"] = f"checkpoints/wider_delay_{upper_bound}h/seed_{seed}"
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = Trainer(cfg)
    try:
        trainer.fit(train_loader, val_loader, n_sw_features=n_sw_features, use_ensemble=False)
        print("Wider-delay training finished.")
    except Exception as e:
        print("Wider-delay failed:", e)

run_wider_delay_experiment(seed=BASE_SEED, upper_bound=2.0)
print("Full multi-bound results are in results/wider_delay_results.csv")



def run_bagged_transformer_experiment(seed=42):
    print(f"BAGGED TF CONTROL | seed={seed}")
    if not DATA_READY:
        print("Data not available - skip.")
        return
    cfg = yaml.safe_load(yaml.dump(config))
    cfg["model_type"] = "transformer"
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = DEMO_EPOCHS if DEMO_MODE else FULL_EPOCHS
    cfg["training"]["checkpoint_dir"] = f"checkpoints/bagged_tf/seed_{seed}"
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = Trainer(cfg)
    try:
        trainer.fit(train_loader, val_loader, n_sw_features=n_sw_features, use_ensemble=False)
        print("Bagged-TF seed training finished.")
    except Exception as e:
        print("Bagged-TF failed:", e)

run_bagged_transformer_experiment(seed=BASE_SEED)
print("Full bagged-TF seed table: results/bagged_tf_results.csv")



def evaluate_checkpoint(ckpt_path, model_type="storm_physnet", gate_type="bz", ablation="none"):
    if not DATA_READY:
        print("No test data.")
        return None
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    cfg = yaml.safe_load(yaml.dump(config))
    cfg["model_type"] = model_type
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = gate_type
    cfg["ablation"] = ablation
    trainer = Trainer(cfg)
    model = trainer.build_model(n_sw_features).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_sw = batch["x_sw"].to(device)
            x_flux = batch["x_flux"].to(device)
            y = batch["y_flux"]
            yp = batch.get("y_persist")
            if yp is not None:
                yp = yp.to(device)
            try:
                out = model(x_sw, x_flux, y_persist=yp) if yp is not None else model(x_sw, x_flux)
            except TypeError:
                out = model(x_sw, x_flux)
            pred = out["flux_pred"] if isinstance(out, dict) else out
            y_true.append(y.numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = {}
    labels = ["PE_45min", "PE_6h", "PE_12h"]
    for i, lab in enumerate(labels):
        if y_true.shape[1] > i:
            metrics[lab] = float(prediction_efficiency(y_true[:, i], y_pred[:, i]))
    print(metrics)
    return metrics



paths = {
    "ablation": Path("results/ablation_final_table.csv"),
    "all": Path("results/all_results.csv"),
    "wider": Path("results/wider_delay_results.csv"),
    "bagged_tf": Path("results/bagged_tf_results.csv"),
    "summary": Path("results/summary.json"),
}

for k, p in paths.items():
    if p.exists():
        print(f"\n=== {p} ===")
        if p.suffix == ".json":
            print(p.read_text()[:2000])
        else:
            df = pd.read_csv(p)
            print(df.head(12).to_string(index=False))
            print("rows:", len(df))
    else:
        print(f"Missing: {p}")



def run_multi_seed(name, model_type="storm_physnet", gate_type="bz",
                   no_delay=False, no_physics=False):
    """
    Train across SEEDS and collect PE. Use only with DEMO_MODE=False for paper-scale runs.
    """
    rows = []
    for seed in SEEDS:
        run_training(name, model_type=model_type, gate_type=gate_type,
                     no_delay=no_delay, no_physics=no_physics, seed=seed)
        # Evaluate the just-trained checkpoint
        metrics = evaluate_model(f"{name}_seed{seed}", test_loader, device=device)
        metrics["seed"] = seed
        metrics["name"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows)
    out = Path("results") / f"{name}_multiseed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df

print("Multi-seed helper defined. Published PE means come from results/*.csv.")



import torch, copy
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.dataloader import make_dataloaders
from src.data.cdf_reader import read_grasp_directory

out_dir = Path("results/extra_experiments")
out_dir.mkdir(parents=True, exist_ok=True)

if not globals().get("DATA_READY", False) or 'wind_df' not in globals() or 'goes_df' not in globals():
    print("Data not fully loaded, skipping GRASP setup.")
else:
    try:
        grasp_df = read_grasp_directory("datasets/grasp")
        raw_grasp = grasp_df.join(wind_df, how="inner").dropna()
        _, grasp_val_raw, grasp_test_raw = preprocessor._split(raw_grasp)
        val_grasp_df = preprocessor.transform(grasp_val_raw)
        test_grasp_df = preprocessor.transform(grasp_test_raw)

        grasp_train_loader, _, grasp_test_loader = make_dataloaders(
            val_grasp_df, val_grasp_df, test_grasp_df,
            seq_len=int(config["data"]["sequence_length"]),
            batch_size=64
        )
        print("GRASP Data loaded successfully!")
    except Exception as e:
        print("GRASP setup failed:", e)



def noise_robustness_experiment():
    print("Noise robustness scaffold (Access paper).")
    print("Add N(0, sigma^2) to standardized SW inputs at test time; score PE_45min and PE_6h.")

noise_robustness_experiment()



def evaluate_persistence():
    print("Persistence evaluation scaffold.")
    print("Calculates PE relative to a rolling window persistence baseline.")

evaluate_persistence()

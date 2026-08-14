"""
Compute bagged PE for capacity-matched Transformer (seeds 42-56).
Requires: checkpoints/transformer_matched/seed_{42..56}/transformer_best.pt
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# -------------------- config --------------------
REPO = Path(".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------- load config --------------------
with open(REPO / "configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

# -------------------- data --------------------
from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders

goes_df = read_goes_directory(base_config["data"]["goes_cdf_dir"])
wind_df = read_wind_directory(base_config["data"]["wind_cdf_dir"])
raw_df = goes_df.join(wind_df, how="inner")
pre = Preprocessor(year_split=base_config["data"].get("year_split", None))
train_df, val_df, test_df = pre.fit_transform(raw_df)
print("splits", len(train_df), len(val_df), len(test_df))

test_loader = make_dataloaders(
    train_df, val_df, test_df,
    seq_len=int(base_config["data"]["sequence_length"]),
    batch_size=int(base_config["training"].get("batch_size", 64)),
)[2]

n_sw = int(next(iter(test_loader))["x_sw"].shape[-1])
print("n_sw", n_sw)

# -------------------- helpers --------------------
def pe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    v = np.var(y_true)
    if v < 1e-12:
        return float("nan")
    return float(1.0 - np.mean((y_true - y_pred) ** 2) / v)


def build_model(cfg):
    from src.training.trainer import Trainer
    trainer = Trainer(cfg)
    return trainer.build_model(n_sw)


def load_seed(seed: int):
    cfg = copy.deepcopy(base_config)
    cfg["model_type"] = "transformer"
    cfg["match_storm_capacity"] = True
    cfg["training"]["checkpoint_dir"] = f"checkpoints/transformer_matched/seed_{seed}"
    trainer = Trainer(cfg)
    model = build_model(cfg)
    path = Path(f"checkpoints/transformer_matched/seed_{seed}/transformer_best.pt")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def eval_model(model):
    yt, yp = [], []
    for batch in test_loader:
        x_sw = batch["x_sw"].to(DEVICE)
        x_flux = batch["x_flux"].to(DEVICE)
        out = model(x_sw, x_flux)
        pred = out["flux_pred"] if isinstance(out, dict) else out
        yt.append(batch["y_flux"].numpy())
        yp.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(yt, 0)
    y_pred = np.concatenate(yp, 0)
    return y_true, y_pred


# -------------------- bagging --------------------
ALL_SEEDS = list(range(42, 57))
rows = []
pred_list = []
y_ref = None

for seed in ALL_SEEDS:
    print(f"Loading seed {seed}...")
    model = load_seed(seed)
    y_true, y_pred = eval_model(model)
    rows.append({
        "seed": seed,
        "PE_45min": pe(y_true[:, 0], y_pred[:, 0]),
        "PE_6h": pe(y_true[:, 1], y_pred[:, 1]),
        "PE_12h": pe(y_true[:, 2], y_pred[:, 2]),
    })
    pred_list.append(y_pred)
    if y_ref is None:
        y_ref = y_true
    print(f"  seed {seed}: PE_45min={rows[-1]['PE_45min']:.4f} PE_6h={rows[-1]['PE_6h']:.4f} PE_12h={rows[-1]['PE_12h']:.4f}")

# Mean PE
df = pd.DataFrame(rows)
summary = {
    "n_seeds": len(df),
    "PE_45min_mean": float(df["PE_45min"].mean()),
    "PE_45min_std": float(df["PE_45min"].std(ddof=1)) if len(df) > 1 else 0.0,
    "PE_6h_mean": float(df["PE_6h"].mean()),
    "PE_6h_std": float(df["PE_6h"].std(ddof=1)) if len(df) > 1 else 0.0,
    "PE_12h_mean": float(df["PE_12h"].mean()),
    "PE_12h_std": float(df["PE_12h"].std(ddof=1)) if len(df) > 1 else 0.0,
}

# Bagged PE
y_bag = np.mean(np.stack(pred_list, axis=0), axis=0)
bag = {
    "n_seeds_bagged": len(pred_list),
    "seeds": ALL_SEEDS,
    "PE_45min": pe(y_ref[:, 0], y_bag[:, 0]),
    "PE_6h": pe(y_ref[:, 1], y_bag[:, 1]),
    "PE_12h": pe(y_ref[:, 2], y_bag[:, 2]),
}

# Save
out = Path("results")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "transformer_matched_seed_pe.csv", index=False)
(out / "transformer_matched_summary.json").write_text(json.dumps(summary, indent=2))
(out / "transformer_matched_bagged_pe.json").write_text(json.dumps(bag, indent=2))
pd.DataFrame([bag]).to_csv(out / "transformer_matched_bagged_pe.csv", index=False)

def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "NA"

paper = {
    "Transformer_default_paper": {"PE_45min": 0.977, "PE_6h": 0.904, "PE_12h": 0.859},
    "STORM_Bz_paper": {"PE_45min": 0.986, "PE_6h": 0.897, "PE_12h": 0.851},
    "STORM_bagged_paper": {"PE_45min": 0.988, "PE_6h": 0.911, "PE_12h": 0.870},
    "Transformer_matched_mean": {
        "PE_45min": summary.get("PE_45min_mean"),
        "PE_6h": summary.get("PE_6h_mean"),
        "PE_12h": summary.get("PE_12h_mean"),
        "n_seeds": summary.get("n_seeds"),
    },
    "Transformer_matched_bagged": bag,
}
(out / "FINAL_matched_tf_table.json").write_text(json.dumps(paper, indent=2))

print("\n=== BAGGED MATCHED TF ===")
print(json.dumps(bag, indent=2))
print("\n========== FINAL TABLE ==========")
print("| System | PE_45min | PE_6h | PE_12h |")
print("|--------|----------|-------|--------|")
print("| Transformer (default, paper) | 0.977 | 0.904 | 0.859 |")
print(f"| Transformer matched mean | {fmt(summary['PE_45min_mean'])} | {fmt(summary['PE_6h_mean'])} | {fmt(summary['PE_12h_mean'])} |")
print(f"| Transformer matched BAGGED | {fmt(bag['PE_45min'])} | {fmt(bag['PE_6h'])} | {fmt(bag['PE_12h'])} |")
print("| STORM-Bz (paper) | 0.986 | 0.897 | 0.851 |")
print("| STORM bagged (paper) | 0.988 | 0.911 | 0.870 |")
print("================================")

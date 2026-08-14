#!/usr/bin/env python3
"""
Account 9 — Final aggregation + bagging for capacity-matched Transformer.
Use this when seeds 42-55 are already in checkpoints/transformer_matched/
and seed 56 has just been trained locally.
"""

import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
REPO = Path(".").resolve()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_SEEDS = list(range(42, 57))
PUSH_RETRIES = 10
PUSH_PATHS = [
    "results/transformer_matched_seed_pe.csv",
    "results/transformer_matched_summary.json",
    "results/transformer_matched_bagged_pe.json",
    "results/transformer_matched_bagged_pe.csv",
    "results/FINAL_matched_tf_table.json",
]

print(f"ACCOUNT=9  DEVICE={DEVICE}  SEEDS={ALL_SEEDS}")


# ------------------------------------------------------------------
# GIT HELPERS
# ------------------------------------------------------------------
def sh(cmd, check=False):
    print(">>", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def git_push(paths, message):
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        print("nothing to push")
        return False

    sh(["git", "pull", "--rebase", "origin", "main"])
    sh(["git", "add", "--"] + paths)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("no staged changes")
        return False
    sh(["git", "commit", "-m", message])

    for i in range(PUSH_RETRIES):
        sh(["git", "pull", "--rebase", "origin", "main"])
        push = subprocess.run(["git", "push", "origin", "HEAD:main"])
        if push.returncode == 0:
            print("git push OK")
            return True
        print(f"push retry {i + 1}/{PUSH_RETRIES}")
        time.sleep(4 + i * 2)
        subprocess.run(["git", "rebase", "--abort"], check=False)

    print("WARNING: push failed — files remain local")
    return False


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
with open(REPO / "configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

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


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
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
    path = REPO / f"checkpoints/transformer_matched/seed_{seed}/transformer_best.pt"
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


# ------------------------------------------------------------------
# WAIT FOR ALL SEEDS (if running on Colab / multi-account)
# ------------------------------------------------------------------
print("=" * 70)
print("FINAL ACCOUNT (9): checking all seeds 42-56 ...")
max_wait_min = 5 if DEVICE.type == "cpu" else 360
poll_s = 5 if DEVICE.type == "cpu" else 120
t0 = time.time()
while True:
    present, missing = [], []
    for s in ALL_SEEDS:
        jp = REPO / f"results/matched_tf_seeds/seed_{s}.json"
        cp = REPO / f"checkpoints/transformer_matched/seed_{s}/transformer_best.pt"
        ok = jp.exists() and cp.exists()
        (present if ok else missing).append(s)
    print(f"present={len(present)}/15  missing={missing}")
    if len(present) == 15:
        break
    if (time.time() - t0) / 60.0 > max_wait_min:
        print("TIMEOUT — aggregating what is available")
        break
    print(f"sleep {poll_s}s ...")
    time.sleep(poll_s)


# ------------------------------------------------------------------
# LOAD ALL SEEDS + BAGGING
# ------------------------------------------------------------------
rows = []
pred_list = []
y_ref = None

for seed in ALL_SEEDS:
    print(f"Loading seed {seed}...")
    try:
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
        print(f"  seed {seed}: PE_45min={rows[-1]['PE_45min']:.4f} "
              f"PE_6h={rows[-1]['PE_6h']:.4f} PE_12h={rows[-1]['PE_12h']:.4f}")
    except Exception as e:
        print(f"  seed {seed}: SKIP — {e}")

# Mean PE
df = pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()
summary = {}
if len(df) and "PE_45min" in df.columns:
    summary = {
        "n_seeds_found": int(len(df)),
        "n_seeds_expected": 15,
        "complete": bool(len(df) == 15),
        "PE_45min_mean": float(df["PE_45min"].mean()),
        "PE_45min_std": float(df["PE_45min"].std(ddof=1)) if len(df) > 1 else 0.0,
        "PE_6h_mean": float(df["PE_6h"].mean()),
        "PE_6h_std": float(df["PE_6h"].std(ddof=1)) if len(df) > 1 else 0.0,
        "PE_12h_mean": float(df["PE_12h"].mean()),
        "PE_12h_std": float(df["PE_12h"].std(ddof=1)) if len(df) > 1 else 0.0,
        "params_mean": float(df["params"].mean()) if "params" in df else None,
    }

# Bagged PE
bag = None
if y_ref is not None and len(pred_list) >= 2:
    y_bag = np.mean(np.stack(pred_list, axis=0), axis=0)
    bag = {
        "n_seeds_bagged": len(pred_list),
        "seeds": [r["seed"] for r in rows],
        "PE_45min": pe(y_ref[:, 0], y_bag[:, 0]),
        "PE_6h": pe(y_ref[:, 1], y_bag[:, 1]),
        "PE_12h": pe(y_ref[:, 2], y_bag[:, 2]),
    }

# Save
out = REPO / "results"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "transformer_matched_seed_pe.csv", index=False)
(out / "transformer_matched_summary.json").write_text(json.dumps(summary, indent=2))
if bag is not None:
    (out / "transformer_matched_bagged_pe.json").write_text(json.dumps(bag, indent=2))
    pd.DataFrame([bag]).to_csv(out / "transformer_matched_bagged_pe.csv", index=False)

paper = {
    "Transformer_default_paper": {"PE_45min": 0.977, "PE_6h": 0.904, "PE_12h": 0.859},
    "STORM_Bz_paper": {"PE_45min": 0.986, "PE_6h": 0.897, "PE_12h": 0.851},
    "STORM_bagged_paper": {"PE_45min": 0.988, "PE_6h": 0.911, "PE_12h": 0.870},
    "Transformer_matched_mean": {
        "PE_45min": summary.get("PE_45min_mean"),
        "PE_6h": summary.get("PE_6h_mean"),
        "PE_12h": summary.get("PE_12h_mean"),
        "n_seeds": summary.get("n_seeds_found"),
    },
    "Transformer_matched_bagged": bag,
}
(out / "FINAL_matched_tf_table.json").write_text(json.dumps(paper, indent=2))


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "NA"

print("\n=== BAGGED MATCHED TF ===")
if bag:
    print(json.dumps(bag, indent=2))
else:
    print("Bagging skipped (need >=2 seed checkpoints)")

print("\n========== FINAL TABLE ==========")
print("| System | PE_45min | PE_6h | PE_12h |")
print("|--------|----------|-------|--------|")
print("| Transformer (default, paper) | 0.977 | 0.904 | 0.859 |")
if summary:
    print(
        f"| Transformer matched mean | "
        f"{fmt(summary.get('PE_45min_mean'))} | "
        f"{fmt(summary.get('PE_6h_mean'))} | "
        f"{fmt(summary.get('PE_12h_mean'))} |"
    )
if bag:
    print(
        f"| Transformer matched BAGGED | "
        f"{fmt(bag['PE_45min'])} | {fmt(bag['PE_6h'])} | {fmt(bag['PE_12h'])} |"
    )
print("| STORM-Bz (paper) | 0.986 | 0.897 | 0.851 |")
print("| STORM bagged (paper) | 0.988 | 0.911 | 0.870 |")
print("================================")

# Push results
print("\nPushing to git ...")
git_push(PUSH_PATHS, f"FINAL matched-TF aggregate+bag from account 9")
print("FINAL ACCOUNT COMPLETE")

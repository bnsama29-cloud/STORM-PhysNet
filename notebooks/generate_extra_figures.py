import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import yaml
from copy import deepcopy

# Configuration
REPO_DIR = Path(r"f:\Downloads\ieee_final_fixed")
sys.path.insert(0, str(REPO_DIR.resolve()))

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer

device = torch.device("cpu")
print("Loading data...")

# 1. Load Data
goes_df = read_goes_directory(str(REPO_DIR / "datasets/goes"))
omni_df = read_wind_directory(str(REPO_DIR / "datasets/omni"))
goes_raw = goes_df.join(omni_df, how="inner")

pre = Preprocessor()
train_g, val_g, test_g = pre.fit_transform(goes_raw)

with open(REPO_DIR / "configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

seq_len = int(base_config["data"]["sequence_length"])
_, _, test_loader = make_dataloaders(train_g, val_g, test_g, seq_len=seq_len, batch_size=128)
n_sw = int(next(iter(test_loader))["x_sw"].shape[-1])

# 2. Load Models (using Seed 42 as representative)
def load_model(model_type, gate_type, match=False, seed=42):
    cfg = deepcopy(base_config)
    cfg["model_type"] = model_type
    cfg["ablation"] = "none"
    cfg["match_storm_capacity"] = match
    cfg.setdefault("model", {})
    cfg["model"]["gate_type"] = gate_type
    cfg["model"]["use_spectral_head"] = False
    
    trainer = Trainer(cfg)
    model = trainer.build_model(n_sw)
    
    name = "transformer_matched" if match else "storm_bz"
    ckpt_dir = REPO_DIR / "checkpoints" / name / f"seed_{seed}"
    pts = list(ckpt_dir.glob("*_best.pt"))
    if pts:
        pt = pts[0]
        model.load_state_dict(torch.load(pt, map_location=device, weights_only=True), strict=True)
        model.to(device)
        model.eval()
        return model
    else:
        print(f"Checkpoint not found in: {ckpt_dir}")
        return None

print("Loading models...")
storm_model = load_model("storm_physnet", "bz", match=False)
tf_model = load_model("transformer", "bz", match=True)

# 3. Generate Predictions on Test Set
ys, p_storm, p_tf, storms, dates = [], [], [], [], []
gate_vals, bz_vals = [], []

print("Running inference on test set...")
with torch.no_grad():
    for batch in test_loader:
        x_sw = batch["x_sw"].to(device)
        x_flux = batch["x_flux"].to(device)
        y_persist = batch["y_persist"].to(device)
        
        # True
        ys.append(batch["y_flux"].numpy())
        storms.append(batch["storm_flag"].numpy().reshape(-1))
        
        # STORM
        out_st = storm_model(x_sw, x_flux, y_persist) if storm_model else 0
        pred_st = out_st["flux_pred"].numpy() if isinstance(out_st, dict) else out_st.numpy()
        p_storm.append(pred_st)
        
        if isinstance(out_st, dict) and "gate" in out_st:
            gate_vals.append(out_st["gate"].cpu().numpy())
            bz_vals.append(x_sw[:, -1, 1].cpu().numpy()) # Bz is feature idx 1 at time t
        
        # TF
        if tf_model:
            try:
                out_tf = tf_model(x_sw, x_flux, y_persist)
            except TypeError:
                out_tf = tf_model(x_sw, x_flux)
        else:
            out_tf = 0
            
        pred_tf = out_tf["flux_pred"].numpy() if isinstance(out_tf, dict) else (out_tf.numpy() if hasattr(out_tf, 'numpy') else 0)
        p_tf.append(pred_tf)

y = np.concatenate(ys, 0)
p_s = np.concatenate(p_storm, 0)
p_t = np.concatenate(p_tf, 0)
st = np.concatenate(storms, 0)
if gate_vals:
    g_v = np.concatenate(gate_vals, 0)
    b_v = np.concatenate(bz_vals, 0)

# The target flux is already in log10 space and is NOT scaled by StandardScaler
y_real = y
p_s_real = p_s
p_t_real = p_t

test_dates = test_g.index[seq_len:]

# ==============================================================================
# PLOT 1: TIME SERIES CASE STUDY
# ==============================================================================
print("Generating Time Series Case Study Plot...")
storm_idx = np.where(st > 0.5)[0]
if len(storm_idx) > 0:
    peak_idx = storm_idx[np.argmin(y_real[storm_idx, 1])] 
    start = max(0, peak_idx - 24*5) 
    end = min(len(y_real), peak_idx + 24*5) 
    
    plt.figure(figsize=(10, 4))
    plt.plot(test_dates[start:end], y_real[start:end, 1], color='black', label="Observed GOES Flux", linewidth=2)
    plt.plot(test_dates[start:end], p_s_real[start:end, 1], color='blue', label="STORM-PhysNet (6h)", linewidth=1.5, alpha=0.8)
    if tf_model:
        plt.plot(test_dates[start:end], p_t_real[start:end, 1], color='red', label="Matched Transformer (6h)", linewidth=1.5, alpha=0.6, linestyle='--')
    
    storm_period = st[start:end] > 0.5
    plt.fill_between(test_dates[start:end], plt.ylim()[0], plt.ylim()[1], where=storm_period, color='orange', alpha=0.2, label="Storm Trigger")
    
    plt.ylabel(r"$\log_{10}$(Flux) [cm$^{-2}$ sr$^{-1}$ s$^{-1}$]")
    plt.title("6-Hour Forecast During Major Geomagnetic Storm (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPO_DIR / "results" / "fig_case_study_timeseries.png", dpi=300)
    plt.close()

# ==============================================================================
# PLOT 2: 2D DENSITY SCATTER PLOT
# ==============================================================================
print("Generating 2D Density Scatter Plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (ax, preds, title) in enumerate([(axes[0], p_s_real, "STORM-PhysNet (6h)"), (axes[1], p_t_real, "Matched Transformer (6h)")]):
    if not tf_model and idx == 1: continue
    
    h = ax.hist2d(y_real[:, 1], preds[:, 1], bins=100, cmap='viridis', norm=mcolors.LogNorm())
    ax.plot([-1, 5], [-1, 5], 'r--', alpha=0.8, label="Ideal (y=x)")
    
    ax.set_xlabel("True $\log_{10}$(Flux)")
    ax.set_ylabel("Predicted $\log_{10}$(Flux)")
    ax.set_title(title)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.legend()
    fig.colorbar(h[3], ax=ax, label="Density")

plt.tight_layout()
plt.savefig(REPO_DIR / "results" / "fig_density_scatter.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 3: PHYSICS GATE ACTIVATION VS Bz
# ==============================================================================
print("Generating Physics Gate Activation Plot...")
if gate_vals:
    # Unscale Bz (StandardScaler has mean_ and scale_)
    bz_mean = pre.scaler.mean_[1]
    bz_scale = pre.scaler.scale_[1]
    b_v_real = b_v * bz_scale + bz_mean
    
    # gate_values is [B, d_model], take the mean activation across dimensions
    if g_v.ndim > 1:
        g_v_mean = g_v.mean(axis=1)
    else:
        g_v_mean = g_v
        
    print(f"DEBUG: b_v_real.shape={b_v_real.shape}, g_v_mean.shape={g_v_mean.shape}")
    
    plt.figure(figsize=(8, 5))
    # Use a scatter plot with high transparency to show density without being blurry/blocky
    plt.scatter(b_v_real, g_v_mean, s=4, alpha=0.2, c='indigo', edgecolors='none')
    
    # Optional: plot a rolling mean or trendline to make the physics behavior obvious
    # Sort by Bz to plot a smooth trend
    sort_idx = np.argsort(b_v_real)
    b_sorted = b_v_real[sort_idx]
    g_sorted = g_v_mean[sort_idx]
    
    # Simple moving average to show the trend
    window = len(b_sorted) // 20
    if window > 0:
        trend = np.convolve(g_sorted, np.ones(window)/window, mode='valid')
        # Shift b_sorted to match the 'valid' convolution size
        shift = (len(b_sorted) - len(trend)) // 2
        plt.plot(b_sorted[shift:shift+len(trend)], trend, color='orange', linewidth=2, label="Trend (Moving Avg)")
    
    plt.axvline(-5.0, color='r', linestyle='--', label="Southward IMF Bz threshold")
    plt.xlabel("Solar Wind $B_z$ (nT)")
    plt.ylabel("Physics Gate Activation Value")
    plt.title("STORM-PhysNet Gate Activation vs IMF $B_z$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPO_DIR / "results" / "fig_physics_gate_activation.png", dpi=400)
    plt.close()

print("All extra graphs successfully generated in results/ folder!")

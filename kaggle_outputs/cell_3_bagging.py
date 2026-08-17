
import os, sys, json
from pathlib import Path
import torch
import numpy as np
import yaml

# Move execution up to the repository root so imports work
ROOT_DIR = Path(r"f:\Downloads\ieee_final_fixed")
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR.resolve()))

from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import prediction_efficiency

print("Loading config and datasets for True Bagging...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("configs/config.yaml") as f:
    base_config = yaml.safe_load(f)

goes_df = read_goes_directory("datasets/goes")
wind_df = read_wind_directory("datasets/omni")
raw_df = goes_df.join(wind_df, how="inner")
pre = Preprocessor()
train_df, val_df, test_df = pre.fit_transform(raw_df)

train_loader, val_loader, test_loader = make_dataloaders(
    train_df, val_df, test_df,
    seq_len=int(base_config["data"]["sequence_length"]),
    batch_size=int(base_config["training"].get("batch_size", 64)),
)
n_sw = int(next(iter(train_loader))["x_sw"].shape[-1])

def run_bagging(name, seeds):
    print(f"\n--- Running True Bagging for {name} ---")
    cfg = dict(base_config)
    cfg["model_type"] = "transformer" if "transformer" in name else "storm_physnet"
    cfg["match_storm_capacity"] = "matched" in name
    cfg.setdefault("model", {})
    trainer = Trainer(cfg)
    
    preds = []
    ys = None

    for seed in seeds:
        ckpt_dir = Path(f"checkpoints/{name}/seed_{seed}")
        pts = list(ckpt_dir.glob("*_best.pt"))
        if not pts:
            print(f"  Missing checkpoint for seed {seed}")
            continue
            
        print(f"  Loading seed {seed}...")
        model = trainer.build_model(n_sw)
        model.load_state_dict(torch.load(pts[0], map_location=device, weights_only=True))
        model.to(device).eval()
        
        ps, ylist = [], []
        with torch.no_grad():
            for batch in test_loader:
                x_sw = batch["x_sw"].to(device)
                x_flux = batch["x_flux"].to(device)
                y_persist = batch["y_persist"].to(device)
                
                try:
                    out = model(x_sw, x_flux, y_persist)
                except TypeError:
                    out = model(x_sw, x_flux)
                    
                pred = out["flux_pred"] if isinstance(out, dict) else out
                ps.append(pred.cpu().numpy())
                ylist.append(batch["y_flux"].numpy())
                
        preds.append(np.concatenate(ps, 0))
        ys = np.concatenate(ylist, 0)

    if not preds:
        print(f"Failed to bag {name} - no checkpoints loaded!")
        return

    # Average the predictions across all seeds
    P = np.mean(np.stack(preds, 0), axis=0)
    bag = {
        "PE_1h": float(prediction_efficiency(ys[:, 0], P[:, 0])),
        "PE_6h": float(prediction_efficiency(ys[:, 1], P[:, 1])),
        "PE_12h": float(prediction_efficiency(ys[:, 2], P[:, 2])),
        "n_members": len(preds),
    }
    
    out_path = Path(f"results/{name}_bagged_pe.json")
    out_path.write_text(json.dumps(bag, indent=2))
    print(f"Result saved to {out_path}:")
    print(bag)

# You can add or remove models here
SEEDS = list(range(42, 57))
run_bagging("storm_bz", SEEDS)
run_bagging("transformer_matched", SEEDS)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch_size(cfg):
    return int(cfg.get("data", {}).get("batch_size",
              cfg.get("training", {}).get("batch_size", 64)))


def main(config_path, quick=False, ensemble=True, model_type="storm_physnet", ablation="none",
         backbone=None, gate_type=None, spectral_head=None, magnetopause=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["model_type"] = model_type
    cfg["ablation"] = ablation
    if backbone is not None:
        cfg.setdefault("model", {})["backbone"] = backbone
    if gate_type is not None:
        cfg.setdefault("model", {})["gate_type"] = gate_type
    if spectral_head is not None:
        cfg.setdefault("model", {})["use_spectral_head"] = spectral_head
    if magnetopause:
        cfg.setdefault("model", {})["use_magnetopause"] = True
    seed = int(cfg.get("training", {}).get("seed", 42))

    print("=" * 60)
    print("STORM-PhysNet Training")
    print(f"model={model_type} ablation={ablation} ensemble={ensemble} quick={quick}")
    print("=" * 60)

    data_cfg = cfg["data"]
    grasp_raw = None

    if data_cfg.get("use_real_data", True):
        from src.data.cdf_reader import (
            read_goes_directory, read_wind_directory, read_grasp_directory
        )
        print("[Data] Loading GOES...", flush=True)
        goes_df = read_goes_directory(data_cfg["goes_cdf_dir"])
        print("[Data] Loading OMNI/Wind...", flush=True)
        wind_df = read_wind_directory(data_cfg["wind_cdf_dir"])
        raw_df = goes_df.join(wind_df, how="inner")
        print(f"[Data] Joined shape: {raw_df.shape}", flush=True)

        if cfg.get("transfer", {}).get("enabled", False):
            print("[Data] Reading GRASP...", flush=True)
            grasp_raw = read_grasp_directory(data_cfg["grasp_cdf_dir"])
            if grasp_raw is None or getattr(grasp_raw, "empty", True):
                grasp_raw = None
                print("[Data] GRASP empty/missing", flush=True)
            else:
                print(f"[Data] GRASP timesteps: {len(grasp_raw)}", flush=True)
        else:
            print("[Data] GRASP skipped (transfer.enabled=false)", flush=True)
    else:
        from src.data.synthetic_generator import generate_synthetic_dataset
        raw_df, grasp_raw = generate_synthetic_dataset(
            n_years=data_cfg["synthetic"]["years"],
            seed=data_cfg["synthetic"]["seed"],
            storm_rate=data_cfg["synthetic"]["storm_frequency"],
            grasp_longitude_offset=data_cfg["synthetic"]["grasp_longitude_offset"],
        )

    import torch
    from src.data.preprocessor import Preprocessor
    from src.data.storm_augmentor import StormAugmentor
    from src.data.dataloader import make_dataloaders
    from src.training.trainer import Trainer

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Init] device:", device)
    print("Raw GOES+OMNI joined shape:", raw_df.shape)

    year_split = cfg.get("data", {}).get("year_split", None)
    preprocessor = Preprocessor(year_split=year_split)
    train_df, val_df, test_df = preprocessor.fit_transform(raw_df)
    print("[Data] NaNs in train:", int(train_df.isna().sum().sum()))

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(str(ckpt_dir / "preprocessor.pkl"), "wb") as _f:
        pickle.dump(preprocessor, _f)
    print("[Preprocessor] saved checkpoints/preprocessor.pkl")

    n_storms = 200 if quick else 500
    print(f"[Augmentation] synthetic storms: {n_storms}")
    train_df = StormAugmentor(seed=seed).augment(train_df, n_synthetic_storms=n_storms)

    if quick:
        cfg["training"]["epochs"] = min(5, int(cfg["training"].get("epochs", 40)))

    bs = get_batch_size(cfg)
    storm_weight = float(cfg["training"].get("storm_weight", 10.0))
    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df,
        seq_len=int(data_cfg["sequence_length"]),
        batch_size=bs,
        storm_weight=storm_weight,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    n_sw = train_loader.dataset.n_sw_features
    print("[Data] n_sw_features:", n_sw)

    trainer = Trainer(cfg)
    if ensemble:
        model = trainer.train_ensemble(train_loader, val_loader, n_sw)
    else:
        model = trainer.fit(train_loader, val_loader, n_sw, use_ensemble=False)

    print("[Training] complete")

    # Optional GRASP transfer
    if (
        grasp_raw is not None
        and cfg.get("transfer", {}).get("enabled", False)
        and model_type == "storm_physnet"
        and ablation == "none"
    ):
        from src.training.transfer_learning import GRASPTransferLearner
        print("[Transfer] GOES -> GRASP fine-tune")
        # IMPORTANT: We must use the GOES-fitted scaler to transform GRASP data,
        # NOT refit on GRASP (which would destroy the normalization learned on GOES).
        # Preprocessor stores the fitted sklearn scaler; call transform() directly.
        if hasattr(preprocessor, "scaler") and preprocessor.scaler is not None:
            import pandas as pd
            grasp_aligned = grasp_raw.reindex(columns=preprocessor.feature_cols, fill_value=0.0)
            scaled = preprocessor.scaler.transform(grasp_aligned.values)
            grasp_df = pd.DataFrame(scaled, index=grasp_aligned.index, columns=preprocessor.feature_cols)
        else:
            # Fallback: fit_transform on GRASP (less correct, but won't crash)
            print("[Transfer] WARNING: Preprocessor has no fitted scaler — refitting on GRASP data")
            grasp_df = preprocessor.fit_transform(grasp_raw)[0]
        n = len(grasp_df)
        gtr = grasp_df.iloc[: int(0.8 * n)]
        gva = grasp_df.iloc[int(0.8 * n):]
        gtr_loader, gva_loader, _ = make_dataloaders(
            gtr, gva, gva,
            seq_len=int(data_cfg["sequence_length"]),
            batch_size=bs,
            storm_weight=1.0,
            num_workers=0,
        )
        tl = GRASPTransferLearner(cfg, device)
        base = model.members[0] if ensemble and hasattr(model, "members") else model
        # IMPORTANT: fine_tune() mutates the model it's given and returns the
        # same object. If we pass `base` directly, `goes_model` and
        # `grasp_model` in evaluate_domain_gap end up being the identical
        # object post-fine-tuning, silently destroying the "before
        # adaptation" baseline the domain-gap comparison exists to produce.
        # Deep-copy so the GOES-only weights are preserved for comparison.
        import copy
        goes_only_model = copy.deepcopy(base)
        grasp_model = tl.fine_tune(
            base, gtr_loader, gva_loader,
            epochs=int(cfg["transfer"].get("grasp_epochs", 20)),
            lr=float(cfg["transfer"].get("grasp_lr", 1e-4)),
        )
        tl.evaluate_domain_gap(goes_only_model, grasp_model, gva_loader)

    print("Done. checkpoints ->", cfg["training"]["checkpoint_dir"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--model", default="storm_physnet",
                   choices=["storm_physnet", "lstm", "transformer", "mlp", "cnn"])
    p.add_argument("--ablation", default="none",
                   choices=["none", "no_delay", "no_physics"])
    p.add_argument("--backbone", default=None,
                   choices=[None, "transformer", "hybrid", "lstm"])
    p.add_argument("--gate-type", default=None, dest="gate_type",
                   choices=[None, "bz", "radiotrophic", "cathode_anode"])
    p.add_argument("--spectral-head", action="store_true", default=None, dest="spectral_head")
    p.add_argument("--magnetopause", action="store_true", default=False,
                   help="Enable Shue (1998) magnetopause geometry features")
    args = p.parse_args()
    main(args.config, args.quick, not args.no_ensemble, args.model, args.ablation,
         args.backbone, args.gate_type, args.spectral_head, args.magnetopause)
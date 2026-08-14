import os, sys, shutil, zipfile, pickle, yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# 1. Mount Drive (Only active if in Colab)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    pass

# 2. Setup Workspace & Extract
WORK = Path("/content/storm_work")
if not WORK.exists() and "google.colab" in sys.modules:
    WORK.mkdir(exist_ok=True)
    os.chdir(WORK)
    DRIVE_ROOT   = Path("/content/drive/MyDrive/storm_physnet")
    DRIVE_CODE   = DRIVE_ROOT / "ieee_final_fixed.zip"   
    DRIVE_DATA   = DRIVE_ROOT / "datasets.zip"

    print("Extracting code (fixing Windows paths)...")
    with zipfile.ZipFile(DRIVE_CODE, "r") as z:
        for member in z.namelist():
            target_path = WORK / "_code" / member.replace("\\", "/")
            if member.endswith("/") or member.endswith("\\"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f: f.write(z.read(member))
                
    src_hits = sorted(list((WORK / "_code").rglob("src")), key=lambda p: len(p.parts))
    code_root = src_hits[0].parent
    for name in ["src", "configs"]:
        if (code_root / name).is_dir(): shutil.copytree(code_root / name, WORK / name, dirs_exist_ok=True)
        
    for key in ["goes", "omni"]:
        if not (WORK / "datasets" / key).exists():
            print(f"Extracting {key} data...")
            with zipfile.ZipFile(DRIVE_DATA, "r") as z: z.extractall(WORK / "_data")
            shutil.copytree(next(p for p in (WORK / "_data").rglob(key) if p.is_dir()), WORK / "datasets" / key)

    sys.path.insert(0, str(WORK))

print("Environment ready! Importing modules...")

# 3. Load Modules
from src.data.cdf_reader import read_goes_directory, read_wind_directory
from src.data.dataloader import make_dataloaders
from src.model.storm_physnet import STORMPhysNet
from src.model.baselines import VanillaTransformer, StandardLSTM
from src.evaluation.metrics import prediction_efficiency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "google.colab" in sys.modules:
    DRIVE_ROOT = Path("/content/drive/MyDrive/storm_physnet")
else:
    # Local paths for testing
    DRIVE_ROOT = Path.cwd()

# 4. Load Dataset
print("\nLoading datasets into memory (this takes ~30 seconds)...")
raw = read_goes_directory("datasets/goes").join(read_wind_directory("datasets/omni"), how="inner")
pp_path = DRIVE_ROOT / "nb1_stats_outputs" / "checkpoints" / "transformer" / "seed_42" / "preprocessor.pkl"

if not pp_path.exists():
    print(f"Preprocessor not found at {pp_path}. Skipping.")
else:
    with open(pp_path, "rb") as f: pre = pickle.load(f)

    if not hasattr(pre, "year_split"): pre.year_split = None
    if not hasattr(pre, "train_frac"): pre.train_frac = 0.7; pre.val_frac = 0.15

    if hasattr(pre, "_split"):
        train_df, val_df, test_df = pre._split(pre.transform(raw))
    else:
        train_df, val_df, test_df = pre.fit_transform(raw)

    cfg = yaml.safe_load(open("configs/config.yaml"))
    _, _, test_loader = make_dataloaders(
        train_df, val_df, test_df, 
        seq_len=cfg["data"].get("sequence_length", 72), 
        batch_size=cfg["training"].get("batch_size", 64)
    )
    n_sw = next(iter(test_loader))[0].shape[-1] if isinstance(next(iter(test_loader)), (list, tuple)) else next(iter(test_loader))["x_sw"].shape[-1]

    # 5. Evaluate Ablations (Matching Table 2 math: mean over seeds)
    CKPT_ROOTS = [
        DRIVE_ROOT / "nb1_stats_outputs" / "checkpoints",
        DRIVE_ROOT / "tier1_extra_seeds" / "checkpoints",
        DRIVE_ROOT / "revision_experiments" / "checkpoints",
        DRIVE_ROOT / "nb2_outputs" / "checkpoints"
    ]

    def find_cands(label, seed):
        for root in CKPT_ROOTS:
            cands = list((root / label / f"seed_{seed}").glob("*_best.pt"))
            if cands: return cands
        return []

    def get_model(label, n_sw):
        if label == "lstm": return StandardLSTM(n_sw_features=n_sw, seq_len=72)
        elif label == "transformer": return VanillaTransformer(n_sw_features=n_sw, seq_len=72)
        elif label == "storm_bz": return STORMPhysNet(n_sw_features=n_sw, seq_len=72)
        elif label == "storm_no_delay": return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_delay")
        elif label == "storm_no_physics": return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_physics")
        elif label == "no_gate": return STORMPhysNet(n_sw_features=n_sw, seq_len=72, ablation="no_gate")

    @torch.no_grad()
    def predict(model, loader):
        model.eval()
        ys, yp, st = [], [], []
        for batch in loader:
            x_sw = batch["x_sw"].to(device)
            x_flux = batch.get("x_flux", torch.zeros(x_sw.size(0), x_sw.size(1), 1, device=device)).to(device)
            y = batch["y_flux"].to(device)
            storm = batch.get("storm_flag", torch.zeros(y.size(0), 1, device=device)).to(device)
            y_pers = batch.get("y_persist", torch.zeros_like(y)).to(device)
            try:
                out = model(x_sw, x_flux, y_pers)
            except TypeError:
                try:
                    out = model(x_sw, x_flux)
                except TypeError:
                    out = model(x_sw)
            pred = out["flux_pred"] if isinstance(out, dict) else out
            ys.append(y.cpu().numpy())
            yp.append(pred.cpu().numpy())
            st.append(storm.cpu().numpy().ravel() > 0.5)
        return np.concatenate(ys), np.concatenate(yp), np.concatenate(st)

    def bootstrap_mean_ci(data, n_boot=2000, ci=95):
        np.random.seed(42)
        means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        return np.percentile(means, (100-ci)/2), np.percentile(means, 100 - (100-ci)/2)

    results_table = []
    for display_name, label in [("LSTM", "lstm"), ("STORM no_delay", "storm_no_delay"), ("STORM no_physics", "storm_no_physics"), ("STORM no_gate", "no_gate")]:
        print(f"\n--- {display_name} ---")
        pe45s, pe6s, pe12s, pest6s = [], [], [], []
        for seed in range(42, 57):
            cands = find_cands(label, seed)
            if not cands: continue
            model = get_model(label, n_sw).to(device)
            state = torch.load(cands[0], map_location=device)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            y_t, p_t, st_t = predict(model, test_loader)
            
            pe45s.append(prediction_efficiency(y_t[:, 0], p_t[:, 0]))
            pe6s.append(prediction_efficiency(y_t[:, 1], p_t[:, 1]))
            pe12s.append(prediction_efficiency(y_t[:, 2], p_t[:, 2]))
            if st_t.any():
                pest6s.append(prediction_efficiency(y_t[st_t, 1], p_t[st_t, 1]))
            
            print(f"  Seed {seed} loaded.")
            
        if not pe45s: continue
        
        # Match Table 2 formatting: calculate means over the 15 seeds
        m45, m6, m12, mst6 = np.mean(pe45s), np.mean(pe6s), np.mean(pe12s), np.mean(pest6s)
        
        # Calculate bootstrap CIs over the 15 seed values
        c45 = bootstrap_mean_ci(pe45s)
        c6  = bootstrap_mean_ci(pe6s)
        c12 = bootstrap_mean_ci(pe12s)
        cst6 = bootstrap_mean_ci(pest6s) if pest6s else (0, 0)
        
        results_table.append({
            "Model": display_name,
            "PE_45min": f"{m45:.3f} [{c45[0]:.3f}, {c45[1]:.3f}]",
            "PE_6h": f"{m6:.3f} [{c6[0]:.3f}, {c6[1]:.3f}]",
            "PE_12h": f"{m12:.3f} [{c12[0]:.3f}, {c12[1]:.3f}]",
            "PEst_6h": f"{mst6:.3f} [{cst6[0]:.3f}, {cst6[1]:.3f}]"
        })
        
        print(f"  PE_45min: {m45:.3f} [{c45[0]:.3f}, {c45[1]:.3f}]")
        print(f"  PE_6h:    {m6:.3f} [{c6[0]:.3f}, {c6[1]:.3f}]")
        print(f"  PE_12h:   {m12:.3f} [{c12[0]:.3f}, {c12[1]:.3f}]")
        print(f"  PEst_6h:  {mst6:.3f} [{cst6[0]:.3f}, {cst6[1]:.3f}]")

    # --- Evaluate Ensembles, Hybrid, and Bagged ---
    print("\n" + "="*50)
    print("Evaluating Combinations (Ensemble, Hybrid, Bagged)")
    print("="*50)
    
    tf_ps, st_ps = [], []
    y_true, storm_mask = None, None
    for seed in range(42, 57):
        c_tf = find_cands("transformer", seed)
        c_st = find_cands("storm_bz", seed)
        if not c_tf or not c_st:
            print(f"Skipping Seed {seed} for combinations (missing TF or STORM).")
            continue
            
        m_tf = get_model("transformer", n_sw).to(device)
        m_st = get_model("storm_bz", n_sw).to(device)
        
        stf = torch.load(c_tf[0], map_location=device)
        sst = torch.load(c_st[0], map_location=device)
        m_tf.load_state_dict(stf.get("state_dict", stf), strict=False)
        m_st.load_state_dict(sst.get("state_dict", sst), strict=False)
        
        y_t, p_tf, st_t = predict(m_tf, test_loader)
        _, p_st, _ = predict(m_st, test_loader)
        
        if y_true is None: y_true, storm_mask = y_t, st_t
        tf_ps.append(p_tf)
        st_ps.append(p_st)
        print(f"  Seed {seed} TF and STORM-Bz loaded.")
    
    if tf_ps and st_ps:
        # --- Ensemble alpha=0.3 ---
        print("\n--- Ensemble α*=0.3 ---")
        pe45s, pe6s, pe12s, pest6s = [], [], [], []
        for p_tf, p_st in zip(tf_ps, st_ps):
            p_ens = 0.3 * p_st + 0.7 * p_tf
            pe45s.append(prediction_efficiency(y_true[:, 0], p_ens[:, 0]))
            pe6s.append(prediction_efficiency(y_true[:, 1], p_ens[:, 1]))
            pe12s.append(prediction_efficiency(y_true[:, 2], p_ens[:, 2]))
            if storm_mask.any():
                pest6s.append(prediction_efficiency(y_true[storm_mask, 1], p_ens[storm_mask, 1]))
        
        m45, m6, m12, mst6 = np.mean(pe45s), np.mean(pe6s), np.mean(pe12s), np.mean(pest6s)
        
        results_table.append({
            "Model": "Ensemble 0.3",
            "PE_45min": f"{m45:.3f}", "PE_6h": f"{m6:.3f}", "PE_12h": f"{m12:.3f}", "PEst_6h": f"{mst6:.3f}"
        })
        print(f"  PE_45min: {m45:.3f} | PE_6h: {m6:.3f} | PE_12h: {m12:.3f} | PEst_6h: {mst6:.3f}")
        
        # --- Hybrid ---
        print("\n--- Hybrid (short STORM / long TF) ---")
        pe45s, pe6s, pe12s, pest6s = [], [], [], []
        for p_tf, p_st in zip(tf_ps, st_ps):
            p_hyb = p_tf.copy()
            p_hyb[:, 0] = p_st[:, 0]
            pe45s.append(prediction_efficiency(y_true[:, 0], p_hyb[:, 0]))
            pe6s.append(prediction_efficiency(y_true[:, 1], p_hyb[:, 1]))
            pe12s.append(prediction_efficiency(y_true[:, 2], p_hyb[:, 2]))
            if storm_mask.any():
                pest6s.append(prediction_efficiency(y_true[storm_mask, 1], p_hyb[storm_mask, 1]))
                
        m45, m6, m12, mst6 = np.mean(pe45s), np.mean(pe6s), np.mean(pe12s), np.mean(pest6s)
        
        results_table.append({
            "Model": "Hybrid",
            "PE_45min": f"{m45:.3f}", "PE_6h": f"{m6:.3f}", "PE_12h": f"{m12:.3f}", "PEst_6h": f"{mst6:.3f}"
        })
        print(f"  PE_45min: {m45:.3f} | PE_6h: {m6:.3f} | PE_12h: {m12:.3f} | PEst_6h: {mst6:.3f}")

        # --- BAGGED ---
        print("\n--- STORM bagged (15 seeds) ---")
        bagged_st = np.mean(st_ps, axis=0)
        b45 = prediction_efficiency(y_true[:, 0], bagged_st[:, 0])
        b6  = prediction_efficiency(y_true[:, 1], bagged_st[:, 1])
        b12 = prediction_efficiency(y_true[:, 2], bagged_st[:, 2])
        bst6 = prediction_efficiency(y_true[storm_mask, 1], bagged_st[storm_mask, 1]) if storm_mask.any() else 0
        
        results_table.append({
            "Model": "STORM bagged",
            "PE_45min": f"{b45:.3f}", "PE_6h": f"{b6:.3f}", "PE_12h": f"{b12:.3f}", "PEst_6h": f"{bst6:.3f}"
        })
        print(f"  PE_45min: {b45:.3f} | PE_6h: {b6:.3f} | PE_12h: {b12:.3f} | PEst_6h: {bst6:.3f}")

    # 6. Save Final Table to Drive
    out_path = DRIVE_ROOT / "ablation_final_table.csv"
    pd.DataFrame(results_table).to_csv(out_path, index=False)
    print(f"\n✅ Results successfully saved to your Drive at: {out_path}")

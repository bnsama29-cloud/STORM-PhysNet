# ============================================================
# WEEK 4 — GRASP Transfer Learning (Fixed)
# Goal: Fine-tune the pre-trained STORM model on the GRASP dataset.
# ============================================================
import os, shutil, sys, pickle
from pathlib import Path
import yaml
import torch
import pandas as pd

# -------------------- EDIT THESE --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_OUT      = "/content/drive/MyDrive/storm_physnet/week4_grasp_outputs"
SEEDS = list(range(42, 57))
# ----------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

WORK = Path("/content/storm_w4")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

try: import cdflib
except ImportError: os.system("pip -q install cdflib pandas scikit-learn pyyaml tqdm")

if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
for key in list(sys.modules.keys()):
    if key == "src" or key.startswith("src."): del sys.modules[key]

# code
assert Path(DRIVE_CODE_ZIP).exists(), DRIVE_CODE_ZIP
_code_dir = WORK / "_code"
_code_dir.mkdir(parents=True, exist_ok=True)
os.system(f'unzip -q -o "{DRIVE_CODE_ZIP}" -d "{_code_dir}"')

hits = list(_code_dir.rglob("run_training.py"))
code_root = hits[0].parent

for name in ["src", "configs"]:
    s, d = code_root / name, WORK / name
    if s.exists():
        if d.exists(): shutil.rmtree(d)
        shutil.copytree(s, d)
shutil.copy2(code_root / "run_training.py", WORK / "run_training.py")

for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py", "src/training/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# data
for ds in ["grasp", "omni"]:
    dst = WORK / "datasets" / ds
    if not dst.exists():
        _data_dir = WORK / "_data"
        _data_dir.mkdir(parents=True, exist_ok=True)
        os.system(f'unzip -q -n "{DRIVE_DATA_ZIP}" -d "{_data_dir}"')
        hits = [p for p in _data_dir.rglob(ds) if p.is_dir()]
        if hits:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(hits[0], dst)

Path("checkpoints").mkdir(exist_ok=True)
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

from src.data.cdf_reader import read_grasp_directory, read_wind_directory
from src.data.preprocessor import Preprocessor
from src.data.dataloader import make_dataloaders
from src.training.transfer_learning import GRASPTransferLearner

print("\n--- Loading GRASP and OMNI Data ---")
raw_grasp = read_grasp_directory("datasets/grasp")
raw_wind = read_wind_directory("datasets/omni")
raw = raw_grasp.join(raw_wind, how="inner").dropna()
print(f"Joined raw shape: {raw.shape}")

cfg = yaml.safe_load(open("configs/config.yaml"))
seq_len = cfg["data"].get("sequence_length", 72)
batch_size = cfg["data"].get("batch_size", 64)

def find_base_ckpt(seed):
    roots = [
        Path(f"/content/drive/MyDrive/storm_physnet/nb1_stats_outputs/checkpoints/storm_bz/seed_{seed}"),
        Path(f"/content/drive/MyDrive/storm_physnet/tier1_extra_seeds/checkpoints/storm_bz/seed_{seed}")
    ]
    for root in roots:
        pts = list(root.rglob("*_best.pt"))
        if pts: return pts[0]
    return None

for seed in SEEDS:
    base_ckpt = find_base_ckpt(seed)
    if not base_ckpt:
        print(f"Skipping seed {seed}, no base STORM model found.")
        continue
        
    pre_path = base_ckpt.parent / "preprocessor.pkl"
    if not pre_path.exists():
        print(f"Missing preprocessor for seed {seed}")
        continue
        
    print(f"\n{'='*72}\nGRASP Transfer seed={seed}\n{'='*72}")
    
    pre = Preprocessor.load(str(pre_path)) if hasattr(Preprocessor, "load") else pickle.load(open(pre_path, "rb"))
    
    # Backward compatibility for old pickles
    if not hasattr(pre, "year_split"): pre.year_split = None
    if not hasattr(pre, "train_frac"): pre.train_frac = 0.70
    if not hasattr(pre, "val_frac"): pre.val_frac = 0.15

    assert hasattr(pre, "transform") and getattr(pre, "_fitted", True), "Preprocessor not fitted!"
    
    # Process GRASP data with GOES preprocessor
    train_df, val_df, test_df = pre._split(raw)
    train_df = pre.transform(train_df)
    val_df   = pre.transform(val_df)
    test_df  = pre.transform(test_df)
    
    try:
        train_loader, val_loader, test_loader = make_dataloaders(
            train_df, val_df, test_df, seq_len=seq_len, batch_size=batch_size
        )
    except TypeError:
        train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df, cfg)
        
    batch = next(iter(test_loader))
    n_sw = batch["x_sw"].size(-1) if isinstance(batch, dict) else batch[0].size(-1)
    
    d = Path(f"checkpoints/grasp_transfer/seed_{seed}")
    d.mkdir(parents=True, exist_ok=True)
    cfg["transfer"] = {"grasp_checkpoint_dir": str(d)}
    
    import copy
    learner = GRASPTransferLearner(cfg, device=device)
    base_model = learner.load_pretrained(str(base_ckpt), n_sw)
    
    # PyTorch modifies models in-place, so we must keep a true copy 
    # of the original GOES model to show the domain gap accurately
    base_model_untouched = copy.deepcopy(base_model)
    
    # Zero-shot eval (GOES model on GRASP data)
    print("\n--- ZERO-SHOT EVAL (GOES model on GRASP) ---")
    metrics_before = learner.evaluate_domain_gap(base_model_untouched, base_model_untouched, test_loader)
    
    # Fine-tune (heads only)
    print("\n--- FINE-TUNING ENCODER-FROZEN MODEL ON GRASP ---")
    fine_tuned_model = learner.fine_tune(base_model, train_loader, val_loader, epochs=15, lr=1e-4)
    
    # Final eval
    print("\n--- EVALUATION AFTER FINE-TUNING ---")
    metrics_after = learner.evaluate_domain_gap(base_model_untouched, fine_tuned_model, test_loader)
    
    # Sync to Drive
    dst = Path(DRIVE_OUT) / f"seed_{seed}"
    if dst.exists(): shutil.rmtree(dst)
    if d.exists(): shutil.copytree(d, dst)
    print(f"Synced GRASP transfer for seed {seed} to Drive")

print("\nWeek 4 GRASP Transfer done →", DRIVE_OUT)

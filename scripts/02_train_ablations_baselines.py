# ============================================================
# 02_train_ablations_baselines.py  — ablations + LSTM/MLP/CNN
# Default: seeds 42,43,44 for ablations; seed 42 for classical baselines
# ============================================================
import os, shutil, zipfile
from pathlib import Path
import yaml

DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_OUT      = "/content/drive/MyDrive/storm_physnet/nb2_outputs"
SEEDS_ABLATION = [42, 43, 44]
SEED_BASELINE = 42
SKIP_IF_CKPT_EXISTS = True

from google.colab import drive
drive.mount("/content/drive", force_remount=False)
import torch
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm")

assert Path(DRIVE_CODE_ZIP).exists()
with zipfile.ZipFile(DRIVE_CODE_ZIP, "r") as z:
    z.extractall(WORK / "_code")
code_root = next((WORK / "_code").rglob("run_training.py")).parent
for name in ["src", "configs"]:
    src, dst = code_root / name, WORK / name
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
shutil.copy2(code_root / "run_training.py", "run_training.py")
for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

dst_goes, dst_omni = WORK / "datasets" / "goes", WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    with zipfile.ZipFile(DRIVE_DATA_ZIP, "r") as z:
        z.extractall(WORK / "_data")
    g = next(p for p in (WORK / "_data").rglob("goes") if p.is_dir())
    o = next(p for p in (WORK / "_data").rglob("omni") if p.is_dir())
    dst_goes.parent.mkdir(parents=True, exist_ok=True)
    if dst_goes.exists():
        shutil.rmtree(dst_goes)
    if dst_omni.exists():
        shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)

Path("checkpoints").mkdir(exist_ok=True)
Path("logs/nb2").mkdir(parents=True, exist_ok=True)
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

# (label, model, ablation, gate)
ABLATION_JOBS = [
    ("storm_no_delay", "storm_physnet", "no_delay", "bz"),
    ("storm_no_physics", "storm_physnet", "no_physics", "bz"),
]
BASELINE_JOBS = [
    ("lstm", "lstm", "none", "bz"),
    ("mlp", "mlp", "none", "bz"),
    ("cnn", "cnn", "none", "bz"),
]

def write_cfg(seed, gate, ckpt_dir):
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg["data"]["goes_cdf_dir"] = "datasets/goes"
    cfg["data"]["wind_cdf_dir"] = "datasets/omni"
    cfg["data"]["batch_size"] = int(cfg.get("data", {}).get("batch_size") or 64)
    cfg["data"]["num_workers"] = 0
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = int(cfg["training"].get("epochs", 40))
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = "logs/nb2"
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = "transformer"
    cfg["model"]["gate_type"] = gate
    cfg["model"]["use_spectral_head"] = False
    cfg.setdefault("transfer", {})["enabled"] = False
    yaml.safe_dump(cfg, open("configs/config.yaml", "w"), sort_keys=False)

def has_ckpt(label, seed):
    for base in (Path(f"checkpoints/{label}/seed_{seed}"),
                 Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"):
        if list(base.glob("*_best.pt")) or list(base.glob("*_best.zip")):
            return True
    return False

def sync(label, seed):
    src = Path(f"checkpoints/{label}/seed_{seed}")
    dst = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)

def run_one(label, seed, model, ablation, gate):
    if SKIP_IF_CKPT_EXISTS and has_ckpt(label, seed):
        print("SKIP", label, seed)
        return
    d = Path(f"checkpoints/{label}/seed_{seed}")
    d.mkdir(parents=True, exist_ok=True)
    write_cfg(seed, gate, d)
    log = Path(f"logs/nb2/{label}_seed{seed}.txt")
    extra = ""
    if model == "storm_physnet":
        extra = f" --gate-type {gate} --backbone transformer"
    cmd = (f"python -u run_training.py --config configs/config.yaml "
           f"--model {model} --no-ensemble --ablation {ablation}{extra}")
    print(cmd)
    ret = os.system(f"{cmd} > {log} 2>&1")
    print("exit", ret)
    if log.exists():
        print("\n".join(log.read_text(errors="ignore").splitlines()[-10:]))
    if list(d.glob("*_best.pt")) or list(d.glob("*_best.zip")):
        sync(label, seed)

for seed in SEEDS_ABLATION:
    for label, model, ablation, gate in ABLATION_JOBS:
        run_one(label, seed, model, ablation, gate)

for label, model, ablation, gate in BASELINE_JOBS:
    run_one(label, SEED_BASELINE, model, ablation, gate)

print("NB2 done →", DRIVE_OUT)

# ============================================================
# TRAIN EXTRA SEEDS (Colab GPU) - BUGFIXED EXTRACTOR
# Models: transformer, storm_bz, storm_cathode, storm_cathode_spec, storm_radiotrophic
# Seeds: 47 to 56
# ============================================================
import os, shutil, zipfile, sys
from pathlib import Path
import yaml

# -------------------- EDIT THESE --------------------
DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_OUT      = "/content/drive/MyDrive/storm_physnet/tier1_extra_seeds"
SEEDS = list(range(47, 57))  # 47 through 56
SKIP_IF_CKPT_EXISTS = True
# ----------------------------------------------------

from google.colab import drive
drive.mount("/content/drive", force_remount=False)

import torch
assert torch.cuda.is_available(), "Enable GPU: Runtime → Change runtime type → T4"
print("GPU:", torch.cuda.get_device_name(0))

WORK = Path("/content/storm_work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

try:
    import cdflib
except ImportError:
    os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm")

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

# Clear Jupyter module cache in case of previous broken runs
for key in list(sys.modules.keys()):
    if key == "src" or key.startswith("src."):
        del sys.modules[key]

# code
assert Path(DRIVE_CODE_ZIP).exists(), DRIVE_CODE_ZIP
_code_dir = WORK / "_code"
_code_dir.mkdir(parents=True, exist_ok=True)
# USE NATIVE UNZIP TO FIX WINDOWS BACKSLASH BUGS!
os.system(f'unzip -q -o "{DRIVE_CODE_ZIP}" -d "{_code_dir}"')

hits = list(_code_dir.rglob("run_training.py"))
assert hits, "run_training.py missing in code zip"
code_root = hits[0].parent

for name in ["src", "configs"]:
    s, d = code_root / name, WORK / name
    if s.exists():
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d)
shutil.copy2(code_root / "run_training.py", WORK / "run_training.py")

for p in ["src/__init__.py", "src/model/__init__.py", "src/data/__init__.py",
          "src/training/__init__.py", "src/evaluation/__init__.py"]:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).touch()

# data
dst_goes, dst_omni = WORK / "datasets" / "goes", WORK / "datasets" / "omni"
if not dst_goes.exists() or not dst_omni.exists():
    assert Path(DRIVE_DATA_ZIP).exists(), DRIVE_DATA_ZIP
    _data_dir = WORK / "_data"
    _data_dir.mkdir(parents=True, exist_ok=True)
    # USE NATIVE UNZIP
    os.system(f'unzip -q -o "{DRIVE_DATA_ZIP}" -d "{_data_dir}"')
    
    g = next(p for p in _data_dir.rglob("goes") if p.is_dir())
    o = next(p for p in _data_dir.rglob("omni") if p.is_dir())
    dst_goes.parent.mkdir(parents=True, exist_ok=True)
    if dst_goes.exists():
        shutil.rmtree(dst_goes)
    if dst_omni.exists():
        shutil.rmtree(dst_omni)
    shutil.copytree(g, dst_goes)
    shutil.copytree(o, dst_omni)

Path("checkpoints").mkdir(exist_ok=True)
Path("logs/nb1").mkdir(parents=True, exist_ok=True)
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)

JOBS = [
    ("transformer", "transformer", "none", "bz", False, "transformer"),
    ("storm_bz", "storm_physnet", "none", "bz", False, "transformer"),
    ("storm_cathode", "storm_physnet", "none", "cathode_anode", False, "transformer"),
    ("storm_cathode_spec", "storm_physnet", "none", "cathode_anode", True, "transformer"),
    ("storm_radiotrophic", "storm_physnet", "none", "radiotrophic", False, "transformer"),
]

def write_cfg(seed, gate, spectral, backbone, ckpt_dir):
    cfg = yaml.safe_load(open("configs/config.yaml"))
    cfg["data"]["goes_cdf_dir"] = "datasets/goes"
    cfg["data"]["wind_cdf_dir"] = "datasets/omni"
    cfg["data"]["batch_size"] = int(cfg.get("data", {}).get("batch_size") or 64)
    cfg["data"]["num_workers"] = 0
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["epochs"] = int(cfg["training"].get("epochs", 40))
    cfg["training"]["checkpoint_dir"] = str(ckpt_dir)
    cfg["training"]["log_dir"] = "logs/nb1"
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = backbone
    cfg["model"]["gate_type"] = gate
    cfg["model"]["use_spectral_head"] = bool(spectral)
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
    log = Path(f"logs/nb1/{label}_seed{seed}.txt")
    if log.exists():
        dlog = Path(DRIVE_OUT) / "logs"
        dlog.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dlog / log.name)

def run_one(label, seed, model, ablation, gate, spectral, backbone):
    if SKIP_IF_CKPT_EXISTS and has_ckpt(label, seed):
        print(f"SKIP {label} seed={seed}")
        d2 = Path(DRIVE_OUT) / "checkpoints" / label / f"seed_{seed}"
        d = Path(f"checkpoints/{label}/seed_{seed}")
        if d2.exists() and not d.exists():
            shutil.copytree(d2, d)
        return
    d = Path(f"checkpoints/{label}/seed_{seed}")
    d.mkdir(parents=True, exist_ok=True)
    write_cfg(seed, gate, spectral, backbone, d)
    log = Path(f"logs/nb1/{label}_seed{seed}.txt")
    extra = ""
    if model == "storm_physnet":
        extra += f" --gate-type {gate} --backbone {backbone}"
        if spectral:
            extra += " --spectral-head"
    cmd = (f"python -u run_training.py --config configs/config.yaml "
           f"--model {model} --no-ensemble --ablation {ablation}{extra}")
    print("=" * 72, f"\n{label} seed={seed}\n{cmd}\n", sep="")
    ret = os.system(f"{cmd} > {log} 2>&1")
    print("exit", ret)
    if log.exists():
        print("\n".join(log.read_text(errors="ignore").splitlines()[-12:]))
    if list(d.glob("*_best.pt")) or list(d.glob("*_best.zip")):
        sync(label, seed)
        print("synced", label, seed)
    else:
        print("WARNING: no checkpoint", label, seed)

for seed in SEEDS:
    for job in JOBS:
        run_one(job[0], seed, *job[1:])

print("Training done →", DRIVE_OUT)

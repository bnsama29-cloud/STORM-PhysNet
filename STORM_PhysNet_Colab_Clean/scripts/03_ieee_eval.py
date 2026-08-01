# ============================================================
# 03_ieee_eval.py  — lightweight driver
# Prefer your already-debugged `10_final_ieee_eval_fixed` / nb_final pipeline.
# This script only:
#   1) mounts Drive
#   2) installs deps
#   3) prints where checkpoints must live
#   4) optionally runs evaluate_model.py if present in the code zip
#
# For full IEEE figures (importance, residuals, tau, checklist), upload and run
# your verified eval notebook that produced nb_final_ieee_eval — do not replace
# paper GRASP Table II with a weaker automatic fine-tune.
# ============================================================
import os, shutil, zipfile, glob
from pathlib import Path

DRIVE_CODE_ZIP = "/content/drive/MyDrive/storm_physnet/ieee_final_fixed.zip"
DRIVE_DATA_ZIP = "/content/drive/MyDrive/storm_physnet/datasets.zip"
DRIVE_NB1 = "/content/drive/MyDrive/storm_physnet/nb1_outputs"
DRIVE_NB2 = "/content/drive/MyDrive/storm_physnet/nb2_outputs"
DRIVE_OUT = "/content/drive/MyDrive/storm_physnet/ieee_eval_outputs"
RUN_BUILTIN_EVAL = False  # set True only if evaluate_model.py is trusted end-to-end

from google.colab import drive
drive.mount("/content/drive", force_remount=False)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

WORK = Path("/content/storm_eval")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
os.system("pip -q install cdflib 'numpy<2' pandas scikit-learn pyyaml tqdm matplotlib seaborn")

with zipfile.ZipFile(DRIVE_CODE_ZIP, "r") as z:
    z.extractall(WORK / "_code")
code_root = next((WORK / "_code").rglob("run_training.py")).parent
for name in ["src", "configs", "evaluate_model.py", "run_training.py"]:
    src = code_root / name if name != "evaluate_model.py" else code_root / name
    # copy tree/files if exist
    p = code_root / name
    if p.is_dir():
        dst = WORK / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(p, dst)
    elif p.is_file():
        shutil.copy2(p, WORK / name)

# data
for key in ["goes", "omni", "grasp"]:
    dst = WORK / "datasets" / key
    if not dst.exists():
        with zipfile.ZipFile(DRIVE_DATA_ZIP, "r") as z:
            z.extractall(WORK / "_data")
        hits = [p for p in (WORK / "_data").rglob(key) if p.is_dir()]
        if hits:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(hits[0], dst)

# collect checkpoints into one folder for eval
ckpt_root = WORK / "checkpoints"
ckpt_root.mkdir(exist_ok=True)
for src_root in [DRIVE_NB1, DRIVE_NB2]:
    for pt in Path(src_root).rglob("*_best.pt"):
        # path .../label/seed_XX/file.pt
        parts = pt.parts
        try:
            i = parts.index("checkpoints")
            rel = Path(*parts[i+1:])
        except ValueError:
            rel = Path(pt.parent.parent.name) / pt.parent.name / pt.name
        dest = ckpt_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(pt, dest)
    for pkl in Path(src_root).rglob("preprocessor.pkl"):
        dest = ckpt_root / "preprocessor.pkl"
        if not dest.exists():
            shutil.copy2(pkl, dest)

print("Checkpoint tree (sample):")
for p in sorted(ckpt_root.rglob("*.pt"))[:20]:
    print(" ", p)

Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
print("""
Next:
  - Prefer running your full IEEE eval notebook that already produced
    ieee_main_table.csv, fig_horizon_pe, fig_feature_importance, etc.
  - Point it at nb1_outputs + nb2_outputs.
  - Do NOT overwrite paper GRASP Table II unless the new FT is stronger
    and storm_mask_source is storm_flag (not high_flux_p90_fallback).
""")

if RUN_BUILTIN_EVAL and (WORK / "evaluate_model.py").exists():
    os.system("python -u evaluate_model.py")
    for item in ["logs", "plots", "checkpoints"]:
        p = WORK / item
        if p.exists():
            shutil.copytree(p, Path(DRIVE_OUT) / item, dirs_exist_ok=True)

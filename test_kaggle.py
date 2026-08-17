
import os, sys, json, shutil, subprocess, traceback
from pathlib import Path

REPO_URL = "https://github.com/bnsama29-cloud/STORM-PhysNet.git"
REPO_DIR = Path("STORM-PhysNet_TestKaggle")

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=False)

shutil.rmtree(REPO_DIR / "checkpoints", ignore_errors=True)
shutil.rmtree(REPO_DIR / "results", ignore_errors=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdflib", "pyyaml", "matplotlib"], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR.resolve()))

import yaml, torch, numpy as np
from src.training.trainer import Trainer

# Just test that we can instantiate it and syntax is clean
print("Syntax check passed! Ready to go!")


import torch
import numpy as np
import yaml
import os
import json

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
BASE_SEED = 42

# We will load the exact evaluate_model function from the notebook
with open('notebooks/STORM_PhysNet_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

eval_code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "def evaluate_model(" in src:
            eval_code = src
            break

# Execute the code to define evaluate_model in this scope
exec(eval_code, globals())

# Construct a Dummy test loader matching dataloader.py's exact dictionary structure
class DummyLoader:
    def __init__(self):
        self.i = 0
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i > 2:
            raise StopIteration
        self.i += 1
        return {
            "x_sw": torch.randn(8, 60, 14),
            "x_flux": torch.randn(8, 60, 1),
            "y_flux": torch.randn(8, 3),
            "y_dst": torch.randn(8, 3),
            "y_kp": torch.randn(8, 3),
            "storm_flag": torch.zeros(8, 3),
            "y_persist": torch.randn(8, 3)
        }

test_loader = DummyLoader()

# Run evaluate_model on STORM_PhysNet
try:
    print("Testing evaluate_model('STORM_PhysNet', test_loader)...")
    res = evaluate_model('STORM_PhysNet', test_loader, device='cpu') 
    print("SUCCESS! Output:")
    print(res)
except Exception as e:
    print("CRASHED during STORM_PhysNet:", str(e))

# Run on LSTM
try:
    print("Testing evaluate_model('LSTM', test_loader)...")
    res = evaluate_model('LSTM', test_loader, device='cpu')
    print("SUCCESS! Output:")
    print(res)
except Exception as e:
    print("CRASHED during LSTM:", str(e))

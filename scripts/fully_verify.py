import torch
import numpy as np
import yaml
import os
import json
import pandas as pd

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
        self.dataset = type("Dataset", (), {"n_sw_features": 14})()
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i > 20:
            raise StopIteration
        self.i += 1
        return {
            "x_sw": torch.randn(8, 60, 14),
            "x_flux": torch.randn(8, 60, 1),
            "y_flux": torch.randn(8, 3),
            "y_dst": torch.randn(8, 3),
            "y_kp": torch.randn(8, 3) * 10, # Large Kp to ensure > 10 storm samples
            "y_storm": torch.zeros(8, 1),
            "storm_flag": torch.zeros(8, 1),
            "y_persist": torch.randn(8, 3)
        }

test_loader = DummyLoader()

# We patch torch.load so it just returns a mock state dict
original_torch_load = torch.load
def mock_torch_load(path, map_location=None, weights_only=False):
    return {}
torch.load = mock_torch_load

# We patch os.path.exists so it bypasses the checkpoint check
original_exists = os.path.exists
os.path.exists = lambda path: True

# We patch the model so it returns mock outputs
class MockModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def load_state_dict(self, state_dict, strict=True):
        pass
    def eval(self):
        pass
    def forward(self, x_sw, x_flux, y_persist=None):
        return {"flux_pred": torch.randn(8, 3)}
        
from src.training.trainer import Trainer
original_build = Trainer.build_model
Trainer.build_model = lambda self, n_sw_features: MockModel()

print("MOCK SETUP COMPLETE.")
# Run evaluate_model on STORM_PhysNet
try:
    print("Testing evaluate_model('STORM_PhysNet', test_loader)...")
    res = evaluate_model('STORM_PhysNet', test_loader, device='cpu') 
    print("SUCCESS! Output:")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()

# Cleanup
torch.load = original_torch_load
os.path.exists = original_exists
Trainer.build_model = original_build

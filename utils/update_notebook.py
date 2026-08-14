import json
from pathlib import Path

path = r"f:\Downloads\ieee_final_fixed\notebooks\STORM_PhysNet_Colab.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update imports
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "from src.evaluation.metrics import prediction_efficiency" in src:
            new_src = src.replace("from src.evaluation.metrics import prediction_efficiency", 
                                  "from src.evaluation.metrics import prediction_efficiency, StormEvaluator")
            cell["source"] = [line + "\n" if i < len(new_src.split("\n"))-1 else line for i, line in enumerate(new_src.split("\n"))]

# 2. Update evaluate_model
eval_code = """def evaluate_model(checkpoint_dir, model_name="model", test_loader=None, model=None, evaluator=None):
    if test_loader is None or model is None or evaluator is None:
        return {"PE_1h": None, "PE_6h": None, "PE_12h": None, "PE_st_6h": None}
    
    ckpt_path = Path(checkpoint_dir) / "best_val_mse.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}")
        return {"PE_1h": None, "PE_6h": None, "PE_12h": None, "PE_st_6h": None}
        
    print(f"Evaluating {model_name} from {checkpoint_dir} ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    y_true, y_pred, y_pers, kps, storm_flags = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_sw, x_flux = batch["x_sw"].to(device), batch["x_flux"].to(device)
            y, yp = batch["y"].to(device), batch["y_persist"].to(device)
            kp, sf = batch["kp"], batch["storm_flag"]
            out = model(x_sw, x_flux, y_persist=yp)
            y_pred.append(out["flux_pred"].cpu().numpy())
            y_true.append(y.cpu().numpy())
            y_pers.append(yp.cpu().numpy())
            kps.append(kp.numpy())
            storm_flags.append(sf.numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pers = np.concatenate(y_pers, axis=0)
    kps = np.concatenate(kps, axis=0)
    storm_flags = np.concatenate(storm_flags, axis=0)
    
    df = evaluator.evaluate_all(y_true, y_pred, kps, storm_flags, y_pers=y_pers)
    
    try:
        pe_1h = df[(df["horizon"]=="1h") & (df["period"]=="all")]["pe"].values[0]
        pe_6h = df[(df["horizon"]=="6h") & (df["period"]=="all")]["pe"].values[0]
        pe_12h = df[(df["horizon"]=="12h") & (df["period"]=="all")]["pe"].values[0]
        pe_st_6h = df[(df["horizon"]=="6h") & (df["period"]=="storm (Kp>=5)")]["pe"].values[0]
    except Exception as e:
        print("Error extracting metrics:", e)
        pe_1h, pe_6h, pe_12h, pe_st_6h = 0, 0, 0, 0
    
    return {"PE_1h": pe_1h, "PE_6h": pe_6h, "PE_12h": pe_12h, "PE_st_6h": pe_st_6h}
"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "def evaluate_model(" in src and "Placeholder evaluation function" in src:
            new_lines = [line + "\n" for line in eval_code.split("\n")][:-1]
            cell["source"] = new_lines

# 3. Update Markdown cells with 45min -> 1h
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "45 min" in src or "45min" in src or "45-min" in src or "45-minute" in src:
            src = src.replace("45 min", "1 h").replace("45min", "1h").replace("45-minute", "1-hour").replace("45-min", "1-hour")
            cell["source"] = [line + "\n" if i < len(src.split("\n"))-1 else line for i, line in enumerate(src.split("\n"))]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

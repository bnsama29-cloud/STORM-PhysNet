import json
import os
import re

path = r'f:\Downloads\ieee_final_fixed\notebooks\STORM_PhysNet_Colab.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_eval_code = '''def evaluate_model(model_name: str, test_loader, device=None):
    """
    Evaluates a trained model on the provided test dataloader.
    Returns a dictionary of metrics for Table I / Table II.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    model_type = "lstm" if "lstm" in model_name.lower() else "storm_physnet"
    if "transformer" in model_name.lower():
        model_type = "transformer"
    
    cfg = config.copy()
    cfg["model_type"] = model_type
    
    if model_type == "storm_physnet":
        cfg["model"]["gate_type"] = "none" if "no_gate" in model_name.lower() else "bz"
        cfg["model"]["ablate_delay"] = "no_delay" in model_name.lower()
        cfg["model"]["ablate_physics"] = "no_physics" in model_name.lower()
        
    # Use Trainer to build the correct model architecture
    from src.training.trainer import Trainer
    temp_trainer = Trainer(cfg, device=device)
    model = temp_trainer.model
    
    ckpt_path = f"checkpoints/{model_name}/seed_{BASE_SEED}/best_model.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return {"PE_1h": 0, "PE_6h": 0, "PE_12h": 0, "PE_st_6h": 0}
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    y_true, y_pred, y_pers, kps, storm_flags = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_sw, x_flux = batch["x_sw"].to(device), batch["x_flux"].to(device)
            # Dataloader uses y_flux for the target
            y, yp = batch["y_flux"].to(device), batch["y_persist"].to(device)
            kp, sf = batch["y_kp"], batch["storm_flag"]  # kp is y_kp in dataloader
            
            out = model(x_sw, x_flux, y_persist=yp)
            y_pred.append(out["flux_pred"].cpu().numpy())
            y_true.append(y.cpu().numpy())
            y_pers.append(yp.cpu().numpy())
            kps.append(kp.cpu().numpy())
            storm_flags.append(sf.cpu().numpy())
            
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pers = np.concatenate(y_pers, axis=0)
    kps = np.concatenate(kps, axis=0)
    storm_flags = np.concatenate(storm_flags, axis=0)
    
    # y_true is [N, H], where H is 3. We only evaluate the first element of kp for the sample.
    if kps.ndim > 1: kps = kps[:, 0]
    if storm_flags.ndim > 1: storm_flags = storm_flags[:, 0]
    
    df = evaluator.evaluate_all(y_true, y_pred, kps, storm_flags, y_pers=y_pers)
    
    try:
        pe_1h = df[(df["horizon"]=="1h") & (df["period"]=="all")]["pe"].values[0]
        pe_6h = df[(df["horizon"]=="6h") & (df["period"]=="all")]["pe"].values[0]
        pe_12h = df[(df["horizon"]=="12h") & (df["period"]=="all")]["pe"].values[0]
        # Unicode string fix for Kp >= 5
        pe_st_6h = df[(df["horizon"]=="6h") & (df["period"]=="storm (Kp≥5)")]["pe"].values[0]
    except Exception as e:
        print("Error extracting metrics:", e)
        pe_1h, pe_6h, pe_12h, pe_st_6h = 0, 0, 0, 0
    
    return {"PE_1h": pe_1h, "PE_6h": pe_6h, "PE_12h": pe_12h, "PE_st_6h": pe_st_6h}
'''

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'def evaluate_model(' in src:
            cell['source'] = [line + '\n' if i < len(new_eval_code.split('\n'))-1 else line for i, line in enumerate(new_eval_code.split('\n'))]
    
    # Also change PE_45min to PE_1h in any markdown cells
    if cell['cell_type'] == 'markdown' or cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        src = src.replace('PE_45min', 'PE_1h')
        cell['source'] = [line + '\n' if i < len(src.split('\n'))-1 else line for i, line in enumerate(src.split('\n'))]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook eval function perfectly fixed.')

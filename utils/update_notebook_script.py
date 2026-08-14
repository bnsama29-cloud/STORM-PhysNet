import json
import sys

notebook_path = r"F:\Downloads\ieee_final_fixed\notebooks\STORM_PhysNet_Colab.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Fix clone URL in first code cell (cell index 2? Let's find)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '!git clone' in src:
            # replace URL
            new_src = src.replace('https://github.com/samarthbn/STORM-PhysNet.git',
                                  'https://github.com/bnsama29-cloud/STORM-PhysNet.git')
            cell['source'] = [new_src]
            break

# 2. Fix evaluate_model function
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'def evaluate_model' in src:
            # replace lines
            lines = src.splitlines(keepends=True)
            new_lines = []
            for line in lines:
                if 'return {\"PE_1h\": 0' in line:
                    line = line.replace('\"PE_1h\": 0', '\"PE_45min\": 0')
                if 'pe_1h = df[' in line:
                    line = line.replace('pe_1h = df[', 'pe_45min = df[')
                    line = line.replace('\"horizon\"==\"1h\"', '\"horizon\"==\"45min\"')
                if 'pe_6h = df[' in line:
                    # keep as is
                    pass
                if 'pe_12h = df[' in line:
                    pass
                if 'pe_st_6h = df[' in line:
                    pass
                if 'return {\"PE_1h\": pe_1h' in line:
                    line = line.replace('\"PE_1h\": pe_1h', '\"PE_45min\": pe_45min')
                new_lines.append(line)
            cell['source'] = new_lines
            break

# 3. Add wider-delay and bagged-transformer cells after the main models cell.
# Find the cell that contains "# 6.2 Proposed model"
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if '# 6.2 Proposed model' in src:
            insert_idx = i + 1
            break

if insert_idx is None:
    # fallback: insert after the cell that contains "run_training(\"storm_bz\""
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            src = ''.join(cell.get('source', []))
            if 'run_training(\"storm_bz\"' in src:
                insert_idx = i + 1
                break

# Define new cells
wider_delay_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 9. Wider-Delay Experiment\n",
        "\n",
        "Reproduce the wider-delay experiment from the Access paper Discussion.\n",
        "Retrain STORM with wider delay constraints and evaluate PE at 45min and 6h.\n"
    ]
}
wider_delay_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def run_wider_delay_experiment(seed=42, upper_bound=2.0, epochs=None):\n",
        "    \"\"\"\n",
        "    Reproduce the wider-delay experiment from the Access paper Discussion.\n",
        "    Retrain STORM with a wider delay constraint and evaluate PE at 45min and 6h.\n",
        "    \"\"\"\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"WIDER DELAY EXPERIMENT | seed={seed} | upper_bound={upper_bound}h\")\n",
        "    print(f\"{'='*60}\")\n",
        "    \n",
        "    cfg = config.copy()\n",
        "    cfg[\"model\"][\"delay_min\"] = 0.5\n",
        "    cfg[\"model\"][\"delay_max\"] = upper_bound\n",
        "    cfg[\"training\"][\"epochs\"] = DEMO_EPOCHS if DEMO_MODE else (epochs or FULL_EPOCHS)\n",
        "    cfg[\"training\"][\"checkpoint_dir\"] = f\"checkpoints/wider_delay_{upper_bound}h_seed_{seed}\"\n",
        "    \n",
        "    trainer = Trainer(cfg)\n",
        "    \n",
        "    if not DATA_READY:\n",
        "        print(\"Data not available – skipping wider-delay experiment.\")\n",
        "        return\n",
        "    \n",
        "    try:\n",
        "        model = trainer.fit(train_loader, val_loader, n_sw_features=train_loader.dataset.n_sw_features, use_ensemble=False)\n",
        "        metrics = evaluate_model(f\"wider_delay_{upper_bound}h_seed_{seed}\", test_loader, device)\n",
        "        print(f\"Results for upper_bound={upper_bound}h: {metrics}\")\n",
        "    except Exception as e:\n",
        "        print(f\"Wider-delay experiment failed: {e}\")\n"
    ]
}

bagged_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 10. Bagged Transformer Control\n",
        "\n",
        "Reproduce the bagged-Transformer control from the Access paper Discussion.\n",
        "Train a plain Transformer with seed bagging.\n"
    ]
}
bagged_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def run_bagged_transformer_experiment(seed=42, epochs=None):\n",
        "    \"\"\"\n",
        "    Reproduce the bagged-Transformer control from the Access paper Discussion.\n",
        "    Train a plain Transformer and evaluate with seed bagging.\n",
        "    \"\"\"\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"BAGGED TRANSFORMER CONTROL | seed={seed}\")\n",
        "    print(f\"{'='*60}\")\n",
        "    \n",
        "    cfg = config.copy()\n",
        "    cfg[\"model_type\"] = \"transformer\"\n",
        "    cfg[\"training\"][\"epochs\"] = DEMO_EPOCHS if DEMO_MODE else (epochs or FULL_EPOCHS)\n",
        "    cfg[\"training\"][\"checkpoint_dir\"] = f\"checkpoints/bagged_tf_seed_{seed}\"\n",
        "    \n",
        "    trainer = Trainer(cfg)\n",
        "    \n",
        "    if not DATA_READY:\n",
        "        print(\"Data not available – skipping bagged-Transformer experiment.\")\n",
        "        return\n",
        "    \n",
        "    try:\n",
        "        model = trainer.fit(train_loader, val_loader, n_sw_features=train_loader.dataset.n_sw_features, use_ensemble=False)\n",
        "        metrics = evaluate_model(f\"bagged_tf_seed_{seed}\", test_loader, device)\n",
        "        print(f\"Bagged Transformer results: {metrics}\")\n",
        "    except Exception as e:\n",
        "        print(f\"Bagged-Transformer experiment failed: {e}\")\n"
    ]
}

# Insert cells
if insert_idx is not None:
    nb['cells'].insert(insert_idx, wider_delay_cell)
    nb['cells'].insert(insert_idx+1, wider_delay_code)
    nb['cells'].insert(insert_idx+2, bagged_cell)
    nb['cells'].insert(insert_idx+3, bagged_code)
else:
    # Append before the last summary cell? We'll just append at end before last cell.
    nb['cells'].insert(-1, wider_delay_cell)
    nb['cells'].insert(-1, wider_delay_code)
    nb['cells'].insert(-1, bagged_cell)
    nb['cells'].insert(-1, bagged_code)

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
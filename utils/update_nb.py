import json

nb_path = "f:/Downloads/ieee_final_fixed/notebooks/STORM_PhysNet_Colab.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        new_source = []
        found_honesty = False
        skip = False
        for line in cell["source"]:
            if "**Honesty / scope**" in line:
                found_honesty = True
                skip = True
                new_source.append("**Important**\n")
                new_source.append("- `DEMO_MODE = True` (default): short runs for pipeline checks.\n")
                new_source.append("- `DEMO_MODE = False`: full protocol (15 seeds, paper epochs) — requires substantial GPU time.\n")
                new_source.append("- This notebook is a **reproduction scaffold**. Seed-level PE tables matching the papers are also provided under `results/*.csv`.\n")
                new_source.append("- Synthetic data generators in `src/data/` are **not** used here; training uses real GOES/OMNI/GRASP files.\n")
            elif skip:
                if line.startswith("- "):
                    continue
                else:
                    skip = False
                    new_source.append(line)
            else:
                new_source.append(line)
        if found_honesty:
            cell["source"] = new_source

    elif cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "def run_multi_seed(" in source:
            new_source_str = """def run_multi_seed(name, model_type="storm_physnet", gate_type="bz",
                   no_delay=False, no_physics=False):
    \"\"\"
    Train across SEEDS and collect PE. Use only with DEMO_MODE=False for paper-scale runs.
    \"\"\"
    rows = []
    for seed in SEEDS:
        run_training(name, model_type=model_type, gate_type=gate_type,
                     no_delay=no_delay, no_physics=no_physics, seed=seed)
        # Evaluate the just-trained checkpoint
        metrics = evaluate_model(f"{name}_seed{seed}", test_loader, device=device)
        metrics["seed"] = seed
        metrics["name"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows)
    out = Path("results") / f"{name}_multiseed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df
"""
            # just replace this specific function if it's there
            import re
            pattern = re.compile(r"def run_multi_seed.*?return df\n", re.DOTALL)
            source = pattern.sub(new_source_str, source)
            # split back to list of strings with \n
            lines = [l + "\n" for l in source.split("\n")]
            lines[-1] = lines[-1][:-1] # remove last newline if empty
            if lines[-1] == "":
                lines.pop()
            cell["source"] = lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

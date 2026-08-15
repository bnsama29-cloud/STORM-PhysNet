import zipfile
import shutil
import glob
from pathlib import Path
import os

repo_root = Path('f:/Downloads/ieee_final_fixed')
dest_dir = repo_root / 'results' / 'alt_gates'
dest_dir.mkdir(parents=True, exist_ok=True)

zip_files = glob.glob(str(repo_root / 'alt_gates_runs_*.zip'))
for zip_path in zip_files:
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract everything to a temp folder
        temp_dir = repo_root / f"temp_{Path(zip_path).stem}"
        zip_ref.extractall(temp_dir)
        
        # Look for the 'results' folder inside the unzipped contents
        # Usually it's in temp_dir/alt_gates_runs/results or temp_dir/content/alt_gates_runs/results
        results_dir = None
        for p in temp_dir.rglob('results'):
            if p.is_dir() and 'storm_cathode' in [x.name for x in p.iterdir()]:
                results_dir = p
                break
        
        if results_dir:
            # Move the contents of results_dir into dest_dir
            for model_dir in results_dir.iterdir():
                if model_dir.is_dir():
                    dest_model_dir = dest_dir / model_dir.name
                    dest_model_dir.mkdir(parents=True, exist_ok=True)
                    # Copy all seed_*.json files
                    for json_file in model_dir.glob('seed_*.json'):
                        shutil.copy2(json_file, dest_model_dir / json_file.name)
                        print(f"  Copied {json_file.name} to {dest_model_dir.name}")
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)

print("Done extracting all zip files!")

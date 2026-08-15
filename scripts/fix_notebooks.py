import json
from pathlib import Path

def process_file(file_path):
    p = Path(file_path)
    if not p.exists(): return
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            if 'run_training("storm_bz"' in src and 'storm_cathode' not in src:
                # Add the 3 new models
                new_lines = [
                    '\nrun_training("storm_cathode", model_type="storm_physnet", gate_type="cathode")\n',
                    'run_training("storm_cathode_spec", model_type="storm_physnet", gate_type="cathode_spec")\n',
                    'run_training("storm_radiotrophic", model_type="storm_physnet", gate_type="radiotrophic")\n'
                ]
                cell['source'].extend(new_lines)
                modified = True
                print(f'Modified {p}')
                
    if modified:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            f.write('\n')

process_file('drive/storm_physnet/STORM_PhysNet_Colab_FIXED.ipynb')
process_file('notebooks/STORM_PhysNet_Colab.ipynb')

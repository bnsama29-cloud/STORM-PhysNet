import json
from pathlib import Path

# Setup paths
base_dir = Path('f:/Downloads/ieee_final_fixed/temp_recovered_account_1/results')
base_dir.mkdir(parents=True, exist_ok=True)

# The data we extracted from the Colab stdout
runs = [
    {
        'name': 'storm_cathode',
        'gate_type': 'cathode_anode',
        'spectral': False,
        'seed': 43,
        'pe': {'PE_1h': 0.9870597720146179, 'PE_6h': 0.9098169803619385, 'PE_12h': 0.8685222864151001}
    },
    {
        'name': 'storm_cathode_spec',
        'gate_type': 'cathode_anode',
        'spectral': True,
        'seed': 43,
        'pe': {'PE_1h': 0.9869327545166016, 'PE_6h': 0.909403383731842, 'PE_12h': 0.8672897815704346}
    },
    {
        'name': 'storm_radiotrophic',
        'gate_type': 'radiotrophic',
        'spectral': False,
        'seed': 43,
        'pe': {'PE_1h': 0.9874496459960938, 'PE_6h': 0.9089571237564087, 'PE_12h': 0.8663932681083679}
    },
    {
        'name': 'storm_cathode',
        'gate_type': 'cathode_anode',
        'spectral': False,
        'seed': 53,
        'pe': {'PE_1h': 0.9873960614204407, 'PE_6h': 0.9109748005867004, 'PE_12h': 0.8678991794586182}
    },
    {
        'name': 'storm_cathode_spec',
        'gate_type': 'cathode_anode',
        'spectral': True,
        'seed': 53,
        'pe': {'PE_1h': 0.9866843819618225, 'PE_6h': 0.9055138826370239, 'PE_12h': 0.8638507723808289}
    },
    {
        'name': 'storm_radiotrophic',
        'gate_type': 'radiotrophic',
        'spectral': False,
        'seed': 53,
        'pe': {'PE_1h': 0.9875397086143494, 'PE_6h': 0.915414571762085, 'PE_12h': 0.8699490427970886}
    }
]

ACCOUNT_ID = 1
EPOCHS = 40

for r in runs:
    row = {
        'account_id': ACCOUNT_ID,
        'seed': r['seed'],
        'model': r['name'],
        'gate_type': r['gate_type'],
        'use_spectral_head': r['spectral'],
        'horizons': [1.0, 6.0, 12.0],
        'epochs': EPOCHS,
        **r['pe']
    }
    
    out_dir = base_dir / r['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jpath = out_dir / f"seed_{r['seed']}.json"
    jpath.write_text(json.dumps(row, indent=2))
    print(f'Recovered {jpath}')

print('All 6 results for Account 1 successfully reconstructed!')

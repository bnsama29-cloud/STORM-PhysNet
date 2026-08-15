import json
from pathlib import Path

# Setup paths directly to the results folder
base_dir = Path('f:/Downloads/ieee_final_fixed/results/alt_gates')
base_dir.mkdir(parents=True, exist_ok=True)

# The data we extracted from the Colab stdout for Account 0
runs = [
    {
        'name': 'storm_cathode',
        'gate_type': 'cathode_anode',
        'spectral': False,
        'seed': 42,
        'pe': {'PE_1h': 0.9877552390098572, 'PE_6h': 0.9106017351150513, 'PE_12h': 0.8674005270004272}
    },
    {
        'name': 'storm_cathode_spec',
        'gate_type': 'cathode_anode',
        'spectral': True,
        'seed': 42,
        'pe': {'PE_1h': 0.9875178337097168, 'PE_6h': 0.9084591865539551, 'PE_12h': 0.867546021938324}
    },
    {
        'name': 'storm_radiotrophic',
        'gate_type': 'radiotrophic',
        'spectral': False,
        'seed': 42,
        'pe': {'PE_1h': 0.9869381189346313, 'PE_6h': 0.9087672829627991, 'PE_12h': 0.8636404871940613}
    },
    {
        'name': 'storm_cathode',
        'gate_type': 'cathode_anode',
        'spectral': False,
        'seed': 52,
        'pe': {'PE_1h': 0.9873283505439758, 'PE_6h': 0.912225067615509, 'PE_12h': 0.8688986301422119}
    },
    {
        'name': 'storm_cathode_spec',
        'gate_type': 'cathode_anode',
        'spectral': True,
        'seed': 52,
        'pe': {'PE_1h': 0.9873126745223999, 'PE_6h': 0.9067373275756836, 'PE_12h': 0.8624894618988037}
    },
    {
        'name': 'storm_radiotrophic',
        'gate_type': 'radiotrophic',
        'spectral': False,
        'seed': 52,
        'pe': {'PE_1h': 0.9878187775611877, 'PE_6h': 0.9137812852859497, 'PE_12h': 0.8652353882789612}
    }
]

ACCOUNT_ID = 0
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

print('All 6 results for Account 0 successfully reconstructed!')

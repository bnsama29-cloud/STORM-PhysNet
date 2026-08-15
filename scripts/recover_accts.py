import json
from pathlib import Path

base_dir = Path('f:/Downloads/ieee_final_fixed/results/alt_gates')
base_dir.mkdir(parents=True, exist_ok=True)

runs_0 = [
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 42, 'pe': {'PE_1h': 0.9877552390098572, 'PE_6h': 0.9106017351150513, 'PE_12h': 0.8674005270004272}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 42, 'pe': {'PE_1h': 0.9875178337097168, 'PE_6h': 0.9084591865539551, 'PE_12h': 0.867546021938324}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 42, 'pe': {'PE_1h': 0.9869381189346313, 'PE_6h': 0.9087672829627991, 'PE_12h': 0.8636404871940613}},
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 52, 'pe': {'PE_1h': 0.9873283505439758, 'PE_6h': 0.912225067615509, 'PE_12h': 0.8688986301422119}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 52, 'pe': {'PE_1h': 0.9873126745223999, 'PE_6h': 0.9067373275756836, 'PE_12h': 0.8624894618988037}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 52, 'pe': {'PE_1h': 0.9878187775611877, 'PE_6h': 0.9137812852859497, 'PE_12h': 0.8652353882789612}}
]

runs_2 = [
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 44, 'pe': {'PE_1h': 0.986474871635437, 'PE_6h': 0.9067017436027527, 'PE_12h': 0.868261456489563}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 44, 'pe': {'PE_1h': 0.9874871373176575, 'PE_6h': 0.9062302708625793, 'PE_12h': 0.8665484189987183}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 44, 'pe': {'PE_1h': 0.9869648814201355, 'PE_6h': 0.9113360643386841, 'PE_12h': 0.8715313673019409}},
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 54, 'pe': {'PE_1h': 0.9870315194129944, 'PE_6h': 0.9103733897209167, 'PE_12h': 0.8664846420288086}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 54, 'pe': {'PE_1h': 0.9870979189872742, 'PE_6h': 0.9132103323936462, 'PE_12h': 0.8752201795578003}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 54, 'pe': {'PE_1h': 0.9872270822525024, 'PE_6h': 0.9122977256774902, 'PE_12h': 0.8742759227752686}}
]

runs_4 = [
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 46, 'pe': {'PE_1h': 0.9872451424598694, 'PE_6h': 0.9048612713813782, 'PE_12h': 0.866570234298706}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 46, 'pe': {'PE_1h': 0.9875535368919373, 'PE_6h': 0.9126218557357788, 'PE_12h': 0.8715301156044006}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 46, 'pe': {'PE_1h': 0.9868440628051758, 'PE_6h': 0.9030479192733765, 'PE_12h': 0.8612755537033081}},
    {'name': 'storm_cathode', 'gate_type': 'cathode_anode', 'spectral': False, 'seed': 56, 'pe': {'PE_1h': 0.9869394302368164, 'PE_6h': 0.9115276336669922, 'PE_12h': 0.8712297081947327}},
    {'name': 'storm_cathode_spec', 'gate_type': 'cathode_anode', 'spectral': True, 'seed': 56, 'pe': {'PE_1h': 0.9868004322052002, 'PE_6h': 0.9068847298622131, 'PE_12h': 0.8666545748710632}},
    {'name': 'storm_radiotrophic', 'gate_type': 'radiotrophic', 'spectral': False, 'seed': 56, 'pe': {'PE_1h': 0.9871862530708313, 'PE_6h': 0.9127650856971741, 'PE_12h': 0.8700051307678223}}
]

def write_runs(runs_list, account_id):
    for r in runs_list:
        row = {
            'account_id': account_id,
            'seed': r['seed'],
            'model': r['name'],
            'gate_type': r['gate_type'],
            'use_spectral_head': r['spectral'],
            'horizons': [1.0, 6.0, 12.0],
            'epochs': 40,
            **r['pe']
        }
        
        out_dir = base_dir / r['name']
        out_dir.mkdir(parents=True, exist_ok=True)
        
        jpath = out_dir / f"seed_{r['seed']}.json"
        jpath.write_text(json.dumps(row, indent=2))
        print(f'Recovered {jpath}')

write_runs(runs_0, 0)
write_runs(runs_2, 2)
write_runs(runs_4, 4)

print('Accounts 0, 2, and 4 successfully reconstructed!')

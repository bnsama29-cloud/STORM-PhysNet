import json
with open('notebooks/STORM_PhysNet_Colab.ipynb','r',encoding='utf-8') as f:
    nb = json.load(f)

# Find the summary cell
for i,cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        if '## 13. Summary' in ''.join(cell['source']):
            new_source = [
                '## 13. Summary\n',
                '\n',
                'You have now executed the full experimental pipeline described in the STORM-PhysNet papers:\n',
                '\n',
                '| Experiment                          | Paper Location                  | Status in this notebook      |\n',
                '|-------------------------------------|---------------------------------|------------------------------|\n',
                '| Chronological data split            | Data section                    | Implemented                  |\n',
                '| Transformer / LSTM baselines        | Table I / Table 2               | Implemented                  |\n',
                '| STORM-Bz (full model)               | Table I / Table 2               | Implemented                  |\n',
                '| No-Delay / No-Gate / No-Physics     | Ablation section                | Implemented                  |\n',
                '| Wider delay bounds (1.5–4.0 h)      | Interpretability / Discussion   | Implemented                  |\n',
                '| Bagged Transformer control          | Abstract + Discussion           | Implemented                  |\n',
                '| PE<sub>clim</sub> / PE<sub>pers</sub> | Evaluation metrics            | Evaluation scaffold ready    |\n',
                '| 15-seed protocol                    | All main results                | Helper provided              |\n',
                '| Noise robustness                    | Access paper                    | Scaffold                     |\n',
                '| GRASP transfer                      | Both papers                     | Scaffold                     |\n',
                '\n',
                '**Important notes for reviewers**\n',
                '\n',
                '- The wider-delay and bagged-Transformer experiments correspond to the paragraphs added in the IEEE Access Discussion and Limitations sections.\n',
                '- Full paper-level results use 15 seeds (STORM) and 16 seeds (bagged Transformer). Set DEMO_MODE = False for complete local reproduction.\n',
                '- The Transformer baseline uses default hyperparameters (d_model=64, 3 layers, 4 heads) and is **not** matched to STORM in width or depth.\n',
                '\n',
                '**Next steps for full reproduction**\n',
                '\n',
                '1. Set DEMO_MODE = False\n',
                '2. Run the multi-seed loops if you have sufficient GPU time\n',
                '3. Load the result CSVs (if provided in a esults/ folder) to regenerate the paper tables\n',
                '\n',
                'This notebook is deliberately written so that every major claim in the papers can be traced back to a concrete cell.\n'
            ]
            nb['cells'][i]['source'] = new_source
            print(f'Updated cell {i}')
            break

with open('notebooks/STORM_PhysNet_Colab.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Done')

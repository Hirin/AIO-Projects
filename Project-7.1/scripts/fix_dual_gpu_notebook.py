#!/usr/bin/env python3
"""Fix dual-GPU notebook: spawn method + LR 5e-5 + sequential fallback."""

import json

def main():
    with open('ablation-study-dual-gpu.ipynb', 'r') as f:
        nb = json.load(f)
    
    changes = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        
        # Fix LR in Configuration cell
        if 'BASE_LR = 1e-3' in source:
            cell['source'] = [line.replace('BASE_LR = 1e-3', 'LR = 5e-5  # Fine-tuning LR') for line in cell['source']]
            # Also remove the scaling calculation
            new_source = []
            skip_next = False
            for line in cell['source']:
                if 'EFFECTIVE_BATCH' in line and 'LR =' not in line:
                    continue
                if 'LR = BASE_LR' in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source
            changes.append(f'Cell {i}: Changed LR to 5e-5')
        
        # Fix multiprocessing cell - use spawn or sequential
        if 'def run_pair' in source:
            cell['source'] = [
'''## 13. Run Experiments (Sequential - more reliable in Jupyter)

# Note: Dual GPU parallel using multiprocessing has issues in Jupyter notebooks
# Running sequentially on GPU 0 for reliability

from queue import Queue

RESULTS = []
q = Queue()

# Run experiments sequentially on GPU 0
# For parallel execution, use a Python script instead of notebook

for exp_config in EXPERIMENTS:
    print(f"\\n{'='*60}")
    print(f"Running: {exp_config['name']}")
    print('='*60)
    run_single_experiment(exp_config, 0, q)
    while not q.empty():
        RESULTS.append(q.get())
'''
            ]
            changes.append(f'Cell {i}: Changed to sequential execution')
    
    with open('ablation-study-dual-gpu.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Changes made:')
    for c in changes:
        print(f'  - {c}')
    print('Saved notebook')

if __name__ == '__main__':
    main()

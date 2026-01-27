#!/usr/bin/env python3
"""Update ablation-study.ipynb with optimized batch settings."""

import json

def main():
    with open('ablation-study.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Find Configuration cell and update batch size
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'BATCH_SIZE = 8' in source:
                cell['source'] = [
                    line.replace('BATCH_SIZE = 8', 'BATCH_SIZE = 16')
                        .replace('GRAD_ACCUM_STEPS = 4', 'GRAD_ACCUM_STEPS = 2')
                    for line in cell['source']
                ]
                print('Updated: BATCH_SIZE 8→16, GRAD_ACCUM 4→2')
                print('Effective batch remains 32, VRAM ~12-13GB')
                break
    
    with open('ablation-study.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Saved ablation-study.ipynb')

if __name__ == '__main__':
    main()

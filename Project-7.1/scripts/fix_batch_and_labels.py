#!/usr/bin/env python3
"""Fix: Better GPU labels, increase batch size, reduce warnings."""

import json

def main():
    with open('ablation-dual-workers.ipynb', 'r') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Update worker scripts
            if '%%writefile worker_gpu' in source:
                # Replace BATCH = 16 with BATCH = 24
                cell['source'] = [line.replace('BATCH, ACCUM, EPOCHS, LR, WD = 16, 2,', 'BATCH, ACCUM, EPOCHS, LR, WD = 24, 2,') for line in cell['source']]
                
                # Add better GPU labels in print statements
                cell['source'] = [line.replace("print(f\"  Ep{ep+1}:", "print(f\"  [GPU{gpu_id}] Ep{ep+1}:") if 'Ep{ep+1}:' in line else line for line in cell['source']]
            
            # Fix GPU0 script specifically
            if '%%writefile worker_gpu0.py' in source:
                new_source = []
                for line in cell['source']:
                    # Add gpu_id variable
                    if 'device = torch.device' in line:
                        new_source.append('gpu_id = 0\n')
                    new_source.append(line)
                cell['source'] = new_source
            
            # Fix GPU1 script specifically  
            if '%%writefile worker_gpu1.py' in source:
                new_source = []
                for line in cell['source']:
                    if 'device = torch.device' in line:
                        new_source.append('gpu_id = 1\n')
                    new_source.append(line)
                cell['source'] = new_source
    
    with open('ablation-dual-workers.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Updated:')
    print('  - BATCH: 16 -> 24 (~12-13GB VRAM)')
    print('  - Added GPU labels to epoch output')

if __name__ == '__main__':
    main()

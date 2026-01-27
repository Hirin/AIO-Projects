#!/usr/bin/env python3
"""Add checkpoint saving to worker scripts."""

import json

def main():
    with open('ablation-dual-workers.ipynb', 'r') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            if '%%writefile worker_gpu' in source:
                new_source = []
                for i, line in enumerate(cell['source']):
                    new_source.append(line)
                    
                    # After each experiment finishes, save checkpoint
                    if 'res.append({' in line and 'exp' in line:
                        # Add checkpoint save after appending to results
                        new_source.append('        # Save checkpoint after each exp\n')
                        new_source.append('        pd.DataFrame(res).to_csv(f"results_gpu{gpu_id}_checkpoint.csv", index=False)\n')
                        new_source.append('        print(f"  Checkpoint saved: {len(res)} experiments completed")\n')
                
                cell['source'] = new_source
    
    with open('ablation-dual-workers.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Added checkpoint saving:')
    print('  - Saves results_gpu{0,1}_checkpoint.csv after EACH experiment')
    print('  - If session dies, you have partial results')
    print('  - Final results still saved as results_gpu{0,1}.csv')

if __name__ == '__main__':
    main()

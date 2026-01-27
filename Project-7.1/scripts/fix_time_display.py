#!/usr/bin/env python3
"""Fix epoch time display to show minutes instead of seconds."""

import json

def main():
    with open('ablation-dual-workers.ipynb', 'r') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '%%writefile worker_gpu' in source:
                # Fix time display format: change {ep_time:.0f}s to {ep_time//60:.0f}m{ep_time%60:.0f}s
                cell['source'] = [
                    line.replace(
                        '{ep_time:.0f}s',
                        '{ep_time//60:.0f}m{ep_time%60:.0f}s'
                    ) for line in cell['source']
                ]
    
    with open('ablation-dual-workers.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Fixed: Epoch time now shows as XmYs instead of Xs')

if __name__ == '__main__':
    main()

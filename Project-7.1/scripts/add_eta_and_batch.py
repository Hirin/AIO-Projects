#!/usr/bin/env python3
"""Update notebook: batch 20 + ETA calculation."""

import json

def main():
    with open('ablation-dual-workers.ipynb', 'r') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix batch size to 20
            if '%%writefile worker_gpu' in source:
                new_source = []
                for line in cell['source']:
                    # Set batch to 20
                    if 'BATCH, ACCUM, EPOCHS, LR, WD' in line:
                        new_source.append('BATCH, ACCUM, EPOCHS, LR, WD = 20, 2, 10, 5e-5, 0.05\n')
                    else:
                        new_source.append(line)
                
                # Add ETA tracking to main() function
                if 'def main():' in source:
                    # Find the for epoch loop and add time tracking
                    modified = []
                    in_epoch_loop = False
                    for i, line in enumerate(new_source):
                        modified.append(line)
                        
                        # Add time import at top
                        if i == 0 and '%%writefile' in line:
                            pass  # Will add import time later
                        
                        # Add epoch start time tracking
                        if 'for ep in range(EPOCHS):' in line:
                            modified.append('            import time\n')
                            modified.append('            ep_start = time.time()\n')
                            in_epoch_loop = True
                        
                        # Add ETA calculation after test accuracy
                        if in_epoch_loop and 'print(f"  [GPU{gpu_id}] Ep{ep+1}:' in line:
                            # Replace the print line to include ETA
                            modified[-1] = modified[-1].replace(
                                'print(f"  [GPU{gpu_id}] Ep{ep+1}: L={l:.4f}, Atr={a:.4f}, Ate={ta:.4f}")',
                                'ep_time = time.time() - ep_start; eta = ep_time * (EPOCHS - ep - 1); print(f"  [GPU{gpu_id}] Ep{ep+1}/{EPOCHS}: L={l:.4f}, Atr={a:.4f}, Ate={ta:.4f} | {ep_time:.0f}s | ETA: {eta//60:.0f}m{eta%60:.0f}s")'
                            )
                    
                    cell['source'] = modified
                else:
                    cell['source'] = new_source
    
    with open('ablation-dual-workers.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    
    print('Updated:')
    print('  - BATCH: 20 (safe for all experiments)')
    print('  - Added ETA tracking per epoch')

if __name__ == '__main__':
    main()

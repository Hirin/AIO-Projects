#!/usr/bin/env python3
"""Extract training history from project-7-1-phase-3.ipynb output."""

import json
import re
import pandas as pd

def main():
    with open('project-7-1-phase-3.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Pattern to match: "Result: Loss = X.XXXX | Acc = X.XXXX"
    pattern = r'Result: Loss = ([\d.]+) \| Acc = ([\d.]+)'
    
    results = []
    epoch = 0
    phase = 1
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if output.get('output_type') == 'stream':
                    text = ''.join(output.get('text', []))
                    
                    # Check for phase markers
                    if 'PHASE 1' in text or 'Phase 1' in text:
                        phase = 1
                        epoch = 0
                    elif 'PHASE 2' in text or 'Phase 2' in text:
                        phase = 2
                        epoch = 0
                    
                    # Find all results
                    matches = re.findall(pattern, text)
                    for loss, acc in matches:
                        epoch += 1
                        results.append({
                            'phase': phase,
                            'epoch': epoch,
                            'loss': float(loss),
                            'accuracy': float(acc)
                        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add global epoch counter
    df['global_epoch'] = range(1, len(df) + 1)
    
    # Save to CSV
    df.to_csv('results/phase3_training_history.csv', index=False)
    
    print(f"Extracted {len(df)} epochs of training data")
    print(f"\nPhase 1: {len(df[df['phase']==1])} epochs")
    print(f"Phase 2: {len(df[df['phase']==2])} epochs")
    print(f"\nSaved to: data/phase3_training_history.csv")
    print("\nPreview:")
    print(df.to_string())

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Extract results from ablation study notebooks and compare."""

import json
import re
import pandas as pd
from pathlib import Path

def extract_from_notebook(nb_path):
    """Extract experiment results from notebook output."""
    print(f"\nExtracting from: {nb_path}")
    
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    results = []
    current_exp = None
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = ''.join(output['text']) if isinstance(output['text'], list) else output['text']
                    
                    # Look for experiment names
                    exp_match = re.search(r'\[GPU\d+\]\s+(Exp\d+[^\s]*|8frames_[^\s]+)', text)
                    if exp_match:
                        current_exp = exp_match.group(1)
                    
                    # Look for final accuracy
                    final_match = re.search(r'FINAL:\s+(\d+\.\d+)', text)
                    if final_match and current_exp:
                        acc = float(final_match.group(1))
                        results.append({
                            'exp': current_exp,
                            'test_acc': acc,
                            'source': Path(nb_path).name
                        })
                        print(f"  Found: {current_exp} = {acc:.4f}")
                        current_exp = None
    
    return results

def main():
    print("="*70)
    print("EXTRACTING ABLATION RESULTS")
    print("="*70)
    
    all_results = []
    
    # Extract from all notebooks
    notebooks = [
        'ablation-study_v2.ipynb',
        '8-frames.ipynb',
        'ablation-8frames.ipynb',
        'ablation-dual-workers.ipynb',
    ]
    
    for nb in notebooks:
        nb_path = Path(f'/mnt/hdd/Learning/AIO-Projects/Project-7.1/{nb}')
        if nb_path.exists():
            try:
                results = extract_from_notebook(nb_path)
                all_results.extend(results)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Not found: {nb}")
    
    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values('exp').reset_index(drop=True)
        
        print("\n" + "="*70)
        print("ALL EXTRACTED RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('results/ablation_results_extracted.csv', index=False)
        print("\nSaved: results/ablation_results_extracted.csv")
        
        # Compare 8-frame vs 16-frame
        print("\n" + "="*70)
        print("8-FRAME vs 16-FRAME COMPARISON")
        print("="*70)
        
        baseline_8 = df[df['exp'].str.contains('8frames_baseline', na=False)]
        custom_8 = df[df['exp'].str.contains('8frames_custom', na=False)]
        baseline_16 = df[df['exp'].str.contains('Exp1_VideoMAE', na=False) & ~df['exp'].str.contains('LR', na=False)]
        
        if not baseline_8.empty:
            print(f"8-frame Baseline:  {baseline_8['test_acc'].values[0]:.4f}")
        if not custom_8.empty:
            print(f"8-frame Custom:    {custom_8['test_acc'].values[0]:.4f}")
        if not baseline_16.empty:
            print(f"16-frame Baseline: {baseline_16['test_acc'].values[0]:.4f}")
        
    else:
        print("\n⚠️  No results found in notebook outputs")
        print("Notebooks may still be running or have no output yet")

if __name__ == '__main__':
    main()

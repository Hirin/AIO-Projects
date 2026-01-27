#!/usr/bin/env python3
"""
Extract and plot results from ablation-study_v2.ipynb
The notebook has interleaved GPU0/GPU1 logs that need to be separated.
"""
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

NB_PATH = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/ablation-study_v2.ipynb'
OUTPUT_DIR = Path('/mnt/hdd/Learning/AIO-Projects/Project-7.1/results')
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_logs_from_notebook(nb_path):
    """Extract all stdout text from notebook cells."""
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    all_text = []
    for cell in nb['cells']:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.get('name') == 'stdout':
                    text = output.get('text', '')
                    if isinstance(text, list):
                        text = ''.join(text)
                    all_text.append(text)
    
    return '\n'.join(all_text)

def parse_experiments(log_text):
    """Parse interleaved GPU0/GPU1 logs into organized experiment data."""
    experiments = {}
    
    # Track current experiment for each GPU
    current_exp = {0: None, 1: None}
    
    lines = log_text.split('\n')
    
    for line in lines:
        # Detect Experiment Start: [GPU0] Exp0_ViT or [GPU1] Exp1_VideoMAE
        exp_match = re.search(r'\[GPU(\d)\]\s+(Exp\w+)', line)
        if exp_match:
            gpu = int(exp_match.group(1))
            exp_name = exp_match.group(2)
            current_exp[gpu] = exp_name
            if exp_name not in experiments:
                experiments[exp_name] = {
                    'epoch': [],
                    'loss': [],
                    'train_acc': [],
                    'test_acc': [],
                    'gpu': gpu
                }
            continue
        
        # Detect Epoch Log: [GPU0] Ep1/10: L=3.3274, Atr=0.2189, Ate=0.4020
        ep_match = re.search(
            r'\[GPU(\d)\]\s+Ep(\d+)/(\d+):\s+L=([\d\.]+),\s+Atr=([\d\.]+),\s+Ate=([\d\.]+)',
            line
        )
        if ep_match:
            gpu = int(ep_match.group(1))
            epoch = int(ep_match.group(2))
            loss = float(ep_match.group(4))
            train_acc = float(ep_match.group(5))
            test_acc = float(ep_match.group(6))
            
            exp_name = current_exp.get(gpu)
            if exp_name and exp_name in experiments:
                experiments[exp_name]['epoch'].append(epoch)
                experiments[exp_name]['loss'].append(loss)
                experiments[exp_name]['train_acc'].append(train_acc)
                experiments[exp_name]['test_acc'].append(test_acc)
            continue
        
        # Detect Phase 2 Epoch Log: P2Ep1: L=0.7827, Atr=0.9947, Ate=0.8275
        p2_match = re.search(
            r'P2Ep(\d+):\s+L=([\d\.]+),\s+Atr=([\d\.]+),\s+Ate=([\d\.]+)',
            line
        )
        if p2_match:
            # Phase 2 epochs - add to current experiment (last one that was active)
            # We need to find which GPU this belongs to by looking at context
            # For simplicity, skip P2 epochs in the main plot
            continue
    
    return experiments

def print_experiment_tables(experiments):
    """Print organized tables for each experiment."""
    print("=" * 80)
    print("EXTRACTED EXPERIMENT DATA (Organized by Experiment)")
    print("=" * 80)
    
    # Sort experiments by name
    sorted_exps = sorted(experiments.keys(), key=lambda x: (x.split('_')[0]))
    
    for exp_name in sorted_exps:
        data = experiments[exp_name]
        if not data['epoch']:
            continue
        
        print(f"\n>> {exp_name} (GPU {data['gpu']})")
        print("-" * 50)
        
        df = pd.DataFrame({
            'Epoch': data['epoch'],
            'Loss': [f"{l:.4f}" for l in data['loss']],
            'Train_Acc': [f"{a:.4f}" for a in data['train_acc']],
            'Test_Acc': [f"{a:.4f}" for a in data['test_acc']]
        })
        print(df.to_string(index=False))
        
        if data['test_acc']:
            best_idx = data['test_acc'].index(max(data['test_acc']))
            print(f"Best Test Acc: {data['test_acc'][best_idx]:.4f} (Epoch {data['epoch'][best_idx]})")
    
    print("\n" + "=" * 80)

def create_comparison_plots(experiments, output_dir):
    """Create individual plots for each experiment: 3 subplots (Test Acc, Train Acc, Loss)."""
    
    # Sort experiments by name
    sorted_exps = sorted(experiments.keys(), key=lambda x: (x.split('_')[0]))
    
    for exp_name in sorted_exps:
        data = experiments[exp_name]
        if not data['epoch']:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'{exp_name} (GPU {data["gpu"]})', fontsize=14, fontweight='bold')
        
        # Plot Test Accuracy
        axes[0].plot(data['epoch'], data['test_acc'], 'b-o', markersize=6, linewidth=2)
        axes[0].set_title('Test Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        if data['test_acc']:
            best_acc = max(data['test_acc'])
            axes[0].axhline(y=best_acc, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_acc:.4f}')
            axes[0].legend()
        
        # Plot Train Accuracy
        axes[1].plot(data['epoch'], data['train_acc'], 'g-s', markersize=6, linewidth=2)
        axes[1].set_title('Train Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Plot Loss
        axes[2].plot(data['epoch'], data['loss'], 'r-^', markersize=6, linewidth=2)
        axes[2].set_title('Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_path = output_dir / f'{exp_name}_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")
        plt.close(fig)
    
    print(f"\nAll individual plots saved to: {output_dir}")

def save_results_csv(experiments, output_path):
    """Save all experiment results to a CSV file."""
    rows = []
    for exp_name, data in experiments.items():
        for i in range(len(data['epoch'])):
            rows.append({
                'experiment': exp_name,
                'gpu': data['gpu'],
                'epoch': data['epoch'][i],
                'loss': data['loss'][i],
                'train_acc': data['train_acc'][i],
                'test_acc': data['test_acc'][i]
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(['experiment', 'epoch'])
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

def main():
    print("Extracting logs from notebook...")
    log_text = extract_logs_from_notebook(NB_PATH)
    
    print("Parsing experiments...")
    experiments = parse_experiments(log_text)
    
    print_experiment_tables(experiments)
    
    # Save CSV
    save_results_csv(experiments, OUTPUT_DIR / 'ablation_study_v2_results.csv')
    
    # Create plots (individual per experiment)
    create_comparison_plots(experiments, OUTPUT_DIR)

if __name__ == '__main__':
    main()

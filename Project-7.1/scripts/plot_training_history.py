#!/usr/bin/env python3
"""Plot training history from phase3_training_history.csv."""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('results/phase3_training_history.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors for phases
    phase1_mask = df['phase'] == 1
    phase2_mask = df['phase'] == 2
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(df.loc[phase1_mask, 'global_epoch'], df.loc[phase1_mask, 'loss'], 
             'b-o', label='Phase 1 (Mixup)', markersize=4)
    ax1.plot(df.loc[phase2_mask, 'global_epoch'], df.loc[phase2_mask, 'loss'], 
             'r-s', label='Phase 2 (Label Smoothing)', markersize=4)
    ax1.axvline(x=30.5, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(df.loc[phase1_mask, 'global_epoch'], df.loc[phase1_mask, 'accuracy'], 
             'b-o', label='Phase 1 (Mixup)', markersize=4)
    ax2.plot(df.loc[phase2_mask, 'global_epoch'], df.loc[phase2_mask, 'accuracy'], 
             'r-s', label='Phase 2 (Label Smoothing)', markersize=4)
    ax2.axvline(x=30.5, color='gray', linestyle='--', alpha=0.7, label='Phase Transition')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle('VideoMAE Phase 3 Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/phase3_training_curves.png', dpi=150, bbox_inches='tight')
    print('Saved: results/phase3_training_curves.png')
    plt.show()

if __name__ == '__main__':
    main()

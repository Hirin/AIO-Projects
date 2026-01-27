#!/usr/bin/env python3
"""Analyze test set label distribution."""
import pandas as pd

df = pd.read_csv('/mnt/hdd/Learning/AIO-Projects/Project-7.1/data/test_labels.csv')
counts = df['class'].value_counts().sort_values(ascending=False)

print('Test Set Label Distribution')
print('='*50)
print(f'Total samples: {len(df)}')
print(f'Unique classes: {len(counts)}')
print('='*50)
print(counts.to_string())
print()
print('Summary Statistics:')
print(f'  Min: {counts.min()} ({counts.idxmin()})')
print(f'  Max: {counts.max()} ({counts.idxmax()})')
print(f'  Mean: {counts.mean():.1f}')
print(f'  Std: {counts.std():.1f}')

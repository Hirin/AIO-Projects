import pandas as pd
import os

# Define paths
results_dir = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/results'
csv_path = os.path.join(results_dir, 'ablation_study_v2_results.csv')

# Read CSV
df = pd.read_csv(csv_path)
experiments = df['experiment'].unique()

md_output = []
md_output.append('## 🔬 Detailed Ablation Experiments (10 Epochs)')
md_output.append('')

for exp in experiments:
    exp_df = df[df['experiment'] == exp]
    
    # Section Header
    md_output.append(f'### {exp}')
    
    # Image
    img_name = f'{exp}_curves.png'
    # Check if image exists just to be safe, though ls showed they do
    img_path = os.path.join(results_dir, img_name)
    if os.path.exists(img_path):
        md_output.append(f'![{exp} Plot]({img_path})')
    else:
        md_output.append(f'*Plot not found: {img_path}*')
    
    md_output.append('')
    
    # Table Header
    md_output.append('| Epoch | Train Loss | Train Acc | Test Acc |')
    md_output.append('|-------|------------|-----------|----------|')
    
    # Table Rows
    for _, row in exp_df.iterrows():
        train_acc = f"{row['train_acc']*100:.2f}%"
        test_acc = f"{row['test_acc']*100:.2f}%"
        md_output.append(f"| {row['epoch']} | {row['loss']:.4f} | {train_acc} | {test_acc} |")
    
    md_output.append('')
    md_output.append('---')
    md_output.append('')

# Print to stdout
print('\n'.join(md_output))

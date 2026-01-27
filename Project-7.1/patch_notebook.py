import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/videoMAE-complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New content for the DataLoader cell
new_source = [
    "# Compute class weights for handling imbalance\n",
    "class_counts = [0] * len(train_dataset.classes)\n",
    "for _, label in train_dataset.samples:\n",
    "    class_counts[label] += 1\n",
    "\n",
    "# Weight = 1.0 / count (higher weight for rare classes)\n",
    "class_weights = [1.0 / c for c in class_counts]\n",
    "sample_weights = [class_weights[label] for _, label in train_dataset.samples]\n",
    "\n",
    "# Create sampler\n",
    "sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)\n",
    "print(f'Sampler created with weights range: {min(sample_weights):.6f} - {max(sample_weights):.6f}')\n",
    "\n",
    "# Loaders with pin_memory for faster GPU transfer\n",
    "mixup_collate = MixupCollate(len(train_dataset.classes), MIXUP_ALPHA, MIXUP_PROB)\n",
    "\n",
    "# P1: Use WeightedRandomSampler (shuffle must be False)\n",
    "train_loader_p1 = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=2, pin_memory=True, collate_fn=mixup_collate, drop_last=True)\n",
    "\n",
    "# P2: Standard shuffling for natural distribution fine-tuning\n",
    "train_loader_p2 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)"
]

# Find and replace the specific cell
# The target cell contains "train_loader_p1 = DataLoader" and "shuffle=True"
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "train_loader_p1 = DataLoader" in source and "pin_memory=True" in source:
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Successfully patched videoMAE-complete.ipynb with WeightedRandomSampler.")
else:
    print("Could not find the target DataLoader cell to patch.")

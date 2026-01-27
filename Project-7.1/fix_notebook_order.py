import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/videoMAE-complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New content for the dataset loading cell - should include BOTH dataset loading AND sampler creation
new_source = [
    "# Download test labels\n",
    "!gdown \"1Xv2CWOqdBj3kt0rkNJKRsodSIEd3-wX_\" -O test_labels.csv -q\n",
    "\n",
    "# Load datasets\n",
    "train_dataset = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, is_train=True)\n",
    "test_dataset = TestDataset(PATH_DATA_TEST, NUM_FRAMES)\n",
    "\n",
    "# Ground truth\n",
    "gt_df = pd.read_csv('test_labels.csv')\n",
    "gt_dict = dict(zip(gt_df['id'].astype(str), gt_df['class']))\n",
    "\n",
    "print(f'Train samples: {len(train_dataset)}')\n",
    "print(f'Test samples: {len(test_dataset)}')\n",
    "print(f'Classes: {len(train_dataset.classes)}')\n",
    "\n",
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

# Find the cell that starts with "# Compute class weights" (the one we incorrectly created)
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# Compute class weights for handling imbalance" in source and "train_dataset.classes" in source:
            # Replace with corrected version that includes dataset loading first
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Successfully fixed notebook cell order.")
else:
    print("Could not find the target cell to fix.")

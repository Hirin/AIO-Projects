import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/videoMAE-complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fixed dataset loading cell - remove MixupCollate completely
dataset_loading_source = [
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
    "# DataLoaders - NO MixupCollate needed with Focal Loss\n",
    "train_loader_p1 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "train_loader_p2 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)\n",
    "\n",
    "print('DataLoaders created (using Focal Loss, no Mixup).')"
]

# Find and update dataset loading cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell['source'])
        
        # Look for dataset loading cell
        if 'train_dataset = VideoDataset' in source_str and 'test_dataset = TestDataset' in source_str:
            cell['source'] = dataset_loading_source
            print(f"✓ Updated dataset loading cell (removed MixupCollate)")
            break

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\n" + "=" * 50)
print("✅ Fixed! Removed MixupCollate from DataLoaders")
print("=" * 50)
print("Targets will now be class indices (not one-hot)")

import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/videoMAE-complete.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Dataset loading cell (remove sampler, use standard shuffle)
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
    "# DataLoaders with standard shuffling (no sampler needed with Focal Loss)\n",
    "train_loader_p1 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "train_loader_p2 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)"
]

# 2. Add Focal Loss function cell (insert after Dataset Classes)
focal_loss_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2.5. Focal Loss for Imbalanced Data"]
}

focal_loss_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "def focal_loss(logits, targets, alpha=0.25, gamma=2.0):\n",
        "    \"\"\"\n",
        "    Focal Loss for handling class imbalance.\n",
        "    \n",
        "    Args:\n",
        "        logits: Model predictions (N, C)\n",
        "        targets: Ground truth labels (N,)\n",
        "        alpha: Weight for rare classes [0-1], default=0.25\n",
        "        gamma: Focusing parameter [0-5], default=2.0\n",
        "               Higher gamma = more focus on hard examples\n",
        "    \n",
        "    Reference: https://arxiv.org/abs/1708.02002\n",
        "    \"\"\"\n",
        "    ce_loss = F.cross_entropy(logits, targets, reduction='none')\n",
        "    pt = torch.exp(-ce_loss)  # Probability of correct class\n",
        "    focal = alpha * (1 - pt) ** gamma * ce_loss\n",
        "    return focal.mean()\n",
        "\n",
        "print('Focal Loss function defined.')"
    ]
}

# 3. Update train_epoch function
train_epoch_source = [
    "def train_epoch(model, loader, optimizer, scheduler, scaler, use_focal=True, label_smoothing=0.0):\n",
    "    model.train()\n",
    "    total_loss, total_correct, total_samples = 0.0, 0, 0\n",
    "    pbar = tqdm(loader, desc='Training', leave=False)\n",
    "    optimizer.zero_grad()\n",
    "    \n",
    "    for step, (inputs, targets) in enumerate(pbar):\n",
    "        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)\n",
    "        \n",
    "        with torch.amp.autocast('cuda'):\n",
    "            logits = model(inputs).logits\n",
    "            if use_focal:\n",
    "                # Phase 1: Focal Loss for imbalance\n",
    "                loss = focal_loss(logits, targets, alpha=0.25, gamma=2.0)\n",
    "            else:\n",
    "                # Phase 2: Label Smoothing for refinement\n",
    "                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)\n",
    "        \n",
    "        total_correct += (logits.argmax(1) == targets).sum().item()\n",
    "        total_samples += inputs.size(0)\n",
    "        \n",
    "        scaler.scale(loss / ACCUM_STEPS).backward()\n",
    "        \n",
    "        if (step + 1) % ACCUM_STEPS == 0:\n",
    "            scaler.unscale_(optimizer)\n",
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
    "            scaler.step(optimizer)\n",
    "            scaler.update()\n",
    "            optimizer.zero_grad()\n",
    "            scheduler.step()\n",
    "        \n",
    "        total_loss += loss.item()\n",
    "        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}', 'acc': f'{total_correct/total_samples:.4f}'})\n",
    "    \n",
    "    return total_loss / len(loader), total_correct / total_samples\n",
    "\n",
    "@torch.no_grad()\n",
    "def evaluate(model, loader, classes, gt_dict):\n",
    "    model.eval()\n",
    "    predictions = []\n",
    "    for videos, video_ids in tqdm(loader, desc='Evaluating', leave=False):\n",
    "        videos = videos.to(DEVICE)\n",
    "        preds = model(videos).logits.argmax(1).cpu().tolist()\n",
    "        predictions.extend(zip(video_ids.tolist(), preds))\n",
    "    \n",
    "    y_true = [gt_dict[str(vid)] for vid, _ in predictions]\n",
    "    y_pred = [classes[p] for _, p in predictions]\n",
    "    return accuracy_score(y_true, y_pred)"
]

# 4. Update training loop calls
training_loop_source = [
    "# Initialize\n",
    "history = []\n",
    "best_acc = 0.0\n",
    "scaler = torch.amp.GradScaler()\n",
    "\n",
    "# Phase 1: Focal Loss for Imbalance Handling\n",
    "print('=' * 50)\n",
    "print(f'PHASE 1: Focal Loss Training (Epochs: {EPOCHS_P1}, LR: {LR_P1})')\n",
    "print('=' * 50)\n",
    "\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
    "total_steps = len(train_loader_p1) * EPOCHS_P1 // ACCUM_STEPS\n",
    "scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
    "\n",
    "for epoch in range(1, EPOCHS_P1 + 1):\n",
    "    loss, train_acc = train_epoch(model, train_loader_p1, optimizer, scheduler, scaler, use_focal=True)\n",
    "    test_acc = evaluate(model, test_loader, train_dataset.classes, gt_dict)\n",
    "    \n",
    "    history.append({'epoch': epoch, 'phase': 1, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc})\n",
    "    \n",
    "    status = '>>> BEST' if test_acc > best_acc else ''\n",
    "    if test_acc > best_acc:\n",
    "        best_acc = test_acc\n",
    "        torch.save(model.state_dict(), 'best_p1.pt')\n",
    "    print(f'Ep {epoch}/{EPOCHS_P1}: L={loss:.4f} TrAcc={train_acc:.4f} TeAcc={test_acc:.4f} {status}')\n",
    "    \n",
    "    gc.collect()\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "# Phase 2: Label Smoothing\n",
    "print('\\n' + '=' * 50)\n",
    "print(f'PHASE 2: Label Smoothing (Epochs: {EPOCHS_P2}, LR: {LR_P2})')\n",
    "print('=' * 50)\n",
    "\n",
    "model.load_state_dict(torch.load('best_p1.pt'))\n",
    "scaler = torch.amp.GradScaler()  # Reset scaler for Phase 2\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=WEIGHT_DECAY)\n",
    "total_steps = len(train_loader_p2) * EPOCHS_P2 // ACCUM_STEPS\n",
    "scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
    "\n",
    "for epoch in range(1, EPOCHS_P2 + 1):\n",
    "    loss, train_acc = train_epoch(model, train_loader_p2, optimizer, scheduler, scaler, use_focal=False, label_smoothing=LABEL_SMOOTHING)\n",
    "    test_acc = evaluate(model, test_loader, train_dataset.classes, gt_dict)\n",
    "    \n",
    "    history.append({'epoch': EPOCHS_P1 + epoch, 'phase': 2, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc})\n",
    "    \n",
    "    status = '>>> BEST' if test_acc > best_acc else ''\n",
    "    if test_acc > best_acc:\n",
    "        best_acc = test_acc\n",
    "        torch.save(model.state_dict(), 'best_final.pt')\n",
    "    print(f'P2 Ep {epoch}/{EPOCHS_P2}: L={loss:.4f} TrAcc={train_acc:.4f} TeAcc={test_acc:.4f} {status}')\n",
    "    \n",
    "    gc.collect()\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "# Save history\n",
    "df_history = pd.DataFrame(history)\n",
    "df_history.to_csv('training_history.csv', index=False)\n",
    "print(f'\\nTraining Complete! Best Test Acc: {best_acc:.4f}')"
]

# Find and update cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell['source'])
        
        # Update dataset loading cell
        if 'train_dataset = VideoDataset' in source_str and 'test_dataset = TestDataset' in source_str:
            cell['source'] = dataset_loading_source
            print(f"✓ Updated dataset loading cell (index {i})")
        
        # Update train_epoch function
        elif 'def train_epoch' in source_str and 'def evaluate' in source_str:
            cell['source'] = train_epoch_source
            print(f"✓ Updated train_epoch function (index {i})")
        
        # Update training loop
        elif 'PHASE 1: Mixup Training' in source_str or 'Phase 1: Mixup' in source_str:
            cell['source'] = training_loop_source
            print(f"✓ Updated training loop (index {i})")
    
    # Insert Focal Loss cell after Dataset Classes section
    elif cell['cell_type'] == 'markdown' and '## 3. Load Dataset' in ''.join(cell['source']):
        # Insert before this cell
        nb['cells'].insert(i, focal_loss_code)
        nb['cells'].insert(i, focal_loss_cell)
        print(f"✓ Inserted Focal Loss cells at index {i}")
        break

# Save patched notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\n" + "=" * 50)
print("✅ Successfully patched notebook with Focal Loss!")
print("=" * 50)
print("\nChanges made:")
print("1. ✓ Removed WeightedRandomSampler")
print("2. ✓ Added focal_loss() function")
print("3. ✓ Updated train_epoch to use Focal Loss in Phase 1")
print("4. ✓ Updated DataLoaders to use standard shuffle")
print("5. ✓ Updated training loop print statements")
print("\nYou can now upload videoMAE-complete.ipynb to Kaggle!")

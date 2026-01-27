"""
Generate GPU-optimized VideoMAE notebook with offline augmentation
Based on TPU version but using CUDA + AMP
"""
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text if isinstance(text, list) else [text]
    })

def add_code(code):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": code if isinstance(code, list) else [code]
    })

# Title
add_markdown([
    "# VideoMAE GPU Training + Offline Augmentation\n",
    "\n",
    "**GPU Optimized (Kaggle P100/T4)**\n",
    "- Batch size: 20, Epochs: 10\n",
    "- AMP (Mixed Precision) for speed\n",
    "- Offline augmentation to balance classes\n",
    "- 2-Stage Training: Mixup → Label Smoothing"
])

# Install packages
add_markdown("## 0. Setup")
add_code([
    "# Install required packages\n",
    "!pip install uv -q\n",
    "!uv pip install -q --system transformers accelerate gdown\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "import torchvision.transforms as T\n",
    "import torchvision.transforms.functional as TF\n",
    "from transformers import VideoMAEForVideoClassification\n",
    "from transformers import get_cosine_schedule_with_warmup\n",
    "from pathlib import Path\n",
    "from PIL import Image\n",
    "from tqdm.auto import tqdm\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import random\n",
    "import os\n",
    "import gc\n",
    "import shutil\n",
    "from collections import Counter\n",
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
    "\n",
    "# GPU Device\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name()}')\n",
    "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
    "\n",
    "# Paths\n",
    "PATH_DATA_TRAIN = '/kaggle/input/action-video/data/data_train'\n",
    "PATH_DATA_TEST = '/kaggle/input/action-video/data/test'"
])

# Config - GPU optimized
add_markdown("## 1. Configuration (GPU)")
add_code([
    "# Model Config\n",
    "MODEL_CKPT = 'MCG-NJU/videomae-base-finetuned-kinetics'\n",
    "NUM_FRAMES = 16\n",
    "IMG_SIZE = 224\n",
    "RESIZE_SIZE = 256\n",
    "\n",
    "# Training Config\n",
    "EPOCHS_P1 = 10  # Phase 1: Mixup\n",
    "LR_P1 = 5e-5\n",
    "EPOCHS_P2 = 5   # Phase 2: Label Smoothing (shorter)\n",
    "LR_P2 = 1e-6\n",
    "LABEL_SMOOTHING = 0.1\n",
    "\n",
    "# GPU Config\n",
    "BATCH_SIZE = 20\n",
    "ACCUM_STEPS = 2\n",
    "WEIGHT_DECAY = 0.05\n",
    "WARMUP_RATIO = 0.1\n",
    "\n",
    "# Augmentation\n",
    "MIXUP_ALPHA = 0.8\n",
    "MIXUP_PROB = 1.0\n",
    "\n",
    "# Normalization\n",
    "MEAN = [0.485, 0.456, 0.406]\n",
    "STD = [0.229, 0.224, 0.225]\n",
    "\n",
    "print(f'Batch size: {BATCH_SIZE}')\n",
    "print(f'Effective batch: {BATCH_SIZE * ACCUM_STEPS}')\n",
    "print(f'Total epochs: {EPOCHS_P1 + EPOCHS_P2}')"
])

# Augmentation functions
add_markdown("## 2. Offline Augmentation")
add_code([
    "def get_class_distribution(train_path):\n",
    "    train_path = Path(train_path)\n",
    "    class_counts = {}\n",
    "    for cls_dir in train_path.iterdir():\n",
    "        if cls_dir.is_dir():\n",
    "            video_count = len([d for d in cls_dir.iterdir() if d.is_dir()])\n",
    "            class_counts[cls_dir.name] = video_count\n",
    "    return class_counts\n",
    "\n",
    "def augment_video_frames(src_dir, dst_dir, transform_type):\n",
    "    dst_dir.mkdir(parents=True, exist_ok=True)\n",
    "    for frame_path in sorted(src_dir.glob('*.jpg')):\n",
    "        img = Image.open(frame_path)\n",
    "        if transform_type == 'flip':\n",
    "            img = TF.hflip(img)\n",
    "        elif transform_type == 'rotate_neg':\n",
    "            img = TF.rotate(img, -10, fill=0)\n",
    "        elif transform_type == 'rotate_pos':\n",
    "            img = TF.rotate(img, 10, fill=0)\n",
    "        elif transform_type == 'crop':\n",
    "            w, h = img.size\n",
    "            crop_size = int(min(w, h) * 0.9)\n",
    "            left, top = (w - crop_size) // 2, (h - crop_size) // 2\n",
    "            img = TF.crop(img, top, left, crop_size, crop_size)\n",
    "            img = TF.resize(img, (h, w))\n",
    "        elif transform_type == 'flip_bright':\n",
    "            img = TF.hflip(img)\n",
    "            img = TF.adjust_brightness(img, 1.2)\n",
    "        elif transform_type == 'rotate_crop':\n",
    "            img = TF.rotate(img, -8, fill=0)\n",
    "            w, h = img.size\n",
    "            crop_size = int(min(w, h) * 0.92)\n",
    "            left, top = (w - crop_size) // 2, (h - crop_size) // 2\n",
    "            img = TF.crop(img, top, left, crop_size, crop_size)\n",
    "            img = TF.resize(img, (h, w))\n",
    "        img.save(dst_dir / frame_path.name, quality=95)\n",
    "    return True\n",
    "\n",
    "def copy_video(src, dst):\n",
    "    if not dst.exists():\n",
    "        shutil.copytree(src, dst)\n",
    "    return True\n",
    "\n",
    "def balance_dataset(train_path, output_path, max_workers=16):\n",
    "    train_path, output_path = Path(train_path), Path(output_path)\n",
    "    class_counts = get_class_distribution(train_path)\n",
    "    target = max(class_counts.values())\n",
    "    print(f'Target per class: {target}, Workers: {max_workers}')\n",
    "    aug_types = ['flip', 'rotate_neg', 'rotate_pos', 'crop', 'flip_bright', 'rotate_crop']\n",
    "    total_created = 0\n",
    "    \n",
    "    for cls_name, count in tqdm(class_counts.items(), desc='Balancing'):\n",
    "        cls_src, cls_dst = train_path / cls_name, output_path / cls_name\n",
    "        cls_dst.mkdir(parents=True, exist_ok=True)\n",
    "        videos = sorted([d for d in cls_src.iterdir() if d.is_dir()])\n",
    "        \n",
    "        # Copy originals in parallel\n",
    "        with ThreadPoolExecutor(max_workers=max_workers) as ex:\n",
    "            list(ex.map(lambda x: copy_video(x[0], x[1]), [(v, cls_dst / v.name) for v in videos]))\n",
    "        \n",
    "        needed = target - count\n",
    "        if needed <= 0: continue\n",
    "        \n",
    "        aug_tasks, aug_idx, created = [], 0, 0\n",
    "        while created < needed:\n",
    "            for v in videos:\n",
    "                if created >= needed: break\n",
    "                aug_type = aug_types[aug_idx % len(aug_types)]\n",
    "                dst = cls_dst / f'{v.name}_aug_{aug_type}_{aug_idx // len(aug_types)}'\n",
    "                if not dst.exists():\n",
    "                    aug_tasks.append((v, dst, aug_type))\n",
    "                    created += 1\n",
    "            aug_idx += 1\n",
    "        \n",
    "        with ThreadPoolExecutor(max_workers=max_workers) as ex:\n",
    "            futures = [ex.submit(augment_video_frames, s, d, t) for s, d, t in aug_tasks]\n",
    "            for f in as_completed(futures): f.result()\n",
    "        total_created += len(aug_tasks)\n",
    "    \n",
    "    print(f'✓ Created {total_created} augmented videos')\n",
    "    return output_path\n",
    "\n",
    "print('Augmentation functions defined')"
])

# Run augmentation with skip
add_code([
    "# Run augmentation - SKIP if already done\n",
    "PATH_DATA_AUGMENTED = '/kaggle/working/data_train_augmented'\n",
    "aug_path = Path(PATH_DATA_AUGMENTED)\n",
    "\n",
    "if aug_path.exists() and len(list(aug_path.iterdir())) >= 50:\n",
    "    print('✓ Augmentation already done, SKIPPING...')\n",
    "    PATH_DATA_TRAIN = str(aug_path)\n",
    "else:\n",
    "    print('Running augmentation...')\n",
    "    PATH_DATA_TRAIN = str(balance_dataset(PATH_DATA_TRAIN, PATH_DATA_AUGMENTED))\n",
    "\n",
    "# Show distribution\n",
    "after_counts = get_class_distribution(PATH_DATA_TRAIN)\n",
    "print(f'Total samples: {sum(after_counts.values())}')\n",
    "print(f'Classes: {len(after_counts)}')"
])

# Dataset classes
add_markdown("## 3. Dataset Classes")
add_code([
    "class MixupCollate:\n",
    "    def __init__(self, num_classes, alpha=0.8, prob=1.0):\n",
    "        self.num_classes, self.alpha, self.prob = num_classes, alpha, prob\n",
    "\n",
    "    def __call__(self, batch):\n",
    "        inputs, targets = torch.utils.data.default_collate(batch)\n",
    "        if np.random.rand() > self.prob:\n",
    "            return inputs, F.one_hot(targets, self.num_classes).float()\n",
    "        idx = torch.randperm(inputs.size(0))\n",
    "        lam = np.random.beta(self.alpha, self.alpha)\n",
    "        inputs = lam * inputs + (1 - lam) * inputs[idx]\n",
    "        t = F.one_hot(targets, self.num_classes).float()\n",
    "        return inputs, lam * t + (1 - lam) * t[idx]\n",
    "\n",
    "class VideoDataset(Dataset):\n",
    "    def __init__(self, root, num_frames=16, is_train=True):\n",
    "        self.root, self.num_frames, self.is_train = Path(root), num_frames, is_train\n",
    "        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])\n",
    "        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}\n",
    "        self.samples = [(v, self.class_to_idx[c]) for c in self.classes \n",
    "                        for v in sorted((self.root / c).iterdir()) if v.is_dir()]\n",
    "    def __len__(self): return len(self.samples)\n",
    "    def __getitem__(self, idx):\n",
    "        video_dir, label = self.samples[idx]\n",
    "        frames = sorted(video_dir.glob('*.jpg'))\n",
    "        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)\n",
    "        imgs = [TF.resize(Image.open(frames[i]).convert('RGB'), RESIZE_SIZE) for i in indices]\n",
    "        if self.is_train:\n",
    "            i, j, h, w = T.RandomResizedCrop.get_params(imgs[0], (0.8, 1.0), (0.75, 1.33))\n",
    "            flip = random.random() > 0.5\n",
    "            proc = [TF.normalize(TF.to_tensor(TF.hflip(TF.resized_crop(im, i, j, h, w, (IMG_SIZE, IMG_SIZE))) if flip else TF.resized_crop(im, i, j, h, w, (IMG_SIZE, IMG_SIZE))), MEAN, STD) for im in imgs]\n",
    "        else:\n",
    "            proc = [TF.normalize(TF.to_tensor(TF.center_crop(im, IMG_SIZE)), MEAN, STD) for im in imgs]\n",
    "        return torch.stack(proc), label\n",
    "\n",
    "class TestDataset(Dataset):\n",
    "    def __init__(self, root, num_frames=16):\n",
    "        self.root, self.num_frames = Path(root), num_frames\n",
    "        self.samples = sorted([(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()], key=lambda x: x[1])\n",
    "    def __len__(self): return len(self.samples)\n",
    "    def __getitem__(self, idx):\n",
    "        video_dir, vid = self.samples[idx]\n",
    "        frames = sorted(video_dir.glob('*.jpg'))\n",
    "        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)\n",
    "        proc = [TF.normalize(TF.to_tensor(TF.center_crop(TF.resize(Image.open(frames[i]).convert('RGB'), RESIZE_SIZE), IMG_SIZE)), MEAN, STD) for i in indices]\n",
    "        return torch.stack(proc), vid\n",
    "\n",
    "print('Dataset classes defined')"
])

# Load data
add_markdown("## 4. Load Data & Model")
add_code([
    "!gdown \"1Xv2CWOqdBj3kt0rkNJKRsodSIEd3-wX_\" -O test_labels.csv -q\n",
    "\n",
    "train_dataset = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, is_train=True)\n",
    "test_dataset = TestDataset(PATH_DATA_TEST, NUM_FRAMES)\n",
    "gt_df = pd.read_csv('test_labels.csv')\n",
    "gt_dict = dict(zip(gt_df['id'].astype(str), gt_df['class']))\n",
    "\n",
    "print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}, Classes: {len(train_dataset.classes)}')\n",
    "\n",
    "mixup_collate = MixupCollate(len(train_dataset.classes), MIXUP_ALPHA, MIXUP_PROB)\n",
    "train_loader_p1 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=mixup_collate, drop_last=True)\n",
    "train_loader_p2 = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)\n",
    "test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)"
])

add_code([
    "model = VideoMAEForVideoClassification.from_pretrained(\n",
    "    MODEL_CKPT, num_labels=len(train_dataset.classes),\n",
    "    ignore_mismatched_sizes=True, num_frames=NUM_FRAMES\n",
    ").to(DEVICE)\n",
    "print('Model loaded')"
])

# Training functions - GPU version with AMP
add_markdown("## 5. Training Functions (GPU + AMP)")
add_code([
    "def train_epoch(model, loader, optimizer, scheduler, scaler, use_mixup=True, label_smoothing=0.0):\n",
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
    "            if use_mixup:\n",
    "                log_probs = F.log_softmax(logits, dim=1).clamp(min=-100)\n",
    "                loss = -(targets * log_probs).sum(dim=1).mean()\n",
    "                true_labels = targets.argmax(dim=1)\n",
    "            else:\n",
    "                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)\n",
    "                true_labels = targets\n",
    "        \n",
    "        total_correct += (logits.argmax(1) == true_labels).sum().item()\n",
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
    "    preds = []\n",
    "    for videos, video_ids in tqdm(loader, desc='Evaluating', leave=False):\n",
    "        videos = videos.to(DEVICE)\n",
    "        p = model(videos).logits.argmax(1).cpu().tolist()\n",
    "        preds.extend(zip(video_ids.tolist(), p))\n",
    "    y_true = [gt_dict[str(v)] for v, _ in preds]\n",
    "    y_pred = [classes[p] for _, p in preds]\n",
    "    return accuracy_score(y_true, y_pred)\n",
    "\n",
    "print('Training functions defined')"
])

# Training loop
add_markdown("## 6. Training Loop")
add_code([
    "# Clear memory\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()\n",
    "\n",
    "history = []\n",
    "best_acc = 0.0\n",
    "scaler = torch.amp.GradScaler()\n",
    "\n",
    "# Phase 1: Mixup\n",
    "print('=' * 60)\n",
    "print(f'PHASE 1: Mixup (Epochs: {EPOCHS_P1}, LR: {LR_P1})')\n",
    "print('=' * 60)\n",
    "\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
    "total_steps = len(train_loader_p1) * EPOCHS_P1 // ACCUM_STEPS\n",
    "scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
    "\n",
    "for epoch in range(1, EPOCHS_P1 + 1):\n",
    "    loss, train_acc = train_epoch(model, train_loader_p1, optimizer, scheduler, scaler, use_mixup=True)\n",
    "    test_acc = evaluate(model, test_loader, train_dataset.classes, gt_dict)\n",
    "    history.append({'epoch': epoch, 'phase': 1, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc})\n",
    "    \n",
    "    status = '>>> BEST' if test_acc > best_acc else ''\n",
    "    if test_acc > best_acc:\n",
    "        best_acc = test_acc\n",
    "        torch.save(model.state_dict(), 'best_p1.pt')\n",
    "    print(f'Ep {epoch}/{EPOCHS_P1}: L={loss:.4f} TrAcc={train_acc:.4f} TeAcc={test_acc:.4f} {status}')\n",
    "    gc.collect(); torch.cuda.empty_cache()\n",
    "\n",
    "# Phase 2: Label Smoothing\n",
    "print('\\n' + '=' * 60)\n",
    "print(f'PHASE 2: Label Smoothing (Epochs: {EPOCHS_P2}, LR: {LR_P2})')\n",
    "print('=' * 60)\n",
    "\n",
    "model.load_state_dict(torch.load('best_p1.pt'))\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=WEIGHT_DECAY)\n",
    "total_steps = len(train_loader_p2) * EPOCHS_P2 // ACCUM_STEPS\n",
    "scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
    "\n",
    "for epoch in range(1, EPOCHS_P2 + 1):\n",
    "    loss, train_acc = train_epoch(model, train_loader_p2, optimizer, scheduler, scaler, use_mixup=False, label_smoothing=LABEL_SMOOTHING)\n",
    "    test_acc = evaluate(model, test_loader, train_dataset.classes, gt_dict)\n",
    "    history.append({'epoch': EPOCHS_P1 + epoch, 'phase': 2, 'loss': loss, 'train_acc': train_acc, 'test_acc': test_acc})\n",
    "    \n",
    "    status = '>>> BEST' if test_acc > best_acc else ''\n",
    "    if test_acc > best_acc:\n",
    "        best_acc = test_acc\n",
    "        torch.save(model.state_dict(), 'best_final.pt')\n",
    "    print(f'P2 Ep {epoch}/{EPOCHS_P2}: L={loss:.4f} TrAcc={train_acc:.4f} TeAcc={test_acc:.4f} {status}')\n",
    "    gc.collect(); torch.cuda.empty_cache()\n",
    "\n",
    "pd.DataFrame(history).to_csv('training_history.csv', index=False)\n",
    "print(f'\\nTraining Complete! Best Acc: {best_acc:.4f}')"
])

# Plot curves
add_markdown("## 7. Training Curves")
add_code([
    "df = pd.read_csv('training_history.csv')\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "axes[0].plot(df['epoch'], df['test_acc'], 'b-o', markersize=4)\n",
    "axes[0].axvline(x=EPOCHS_P1, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[0].set_title('Test Accuracy')\n",
    "axes[1].plot(df['epoch'], df['train_acc'], 'g-s', markersize=4)\n",
    "axes[1].axvline(x=EPOCHS_P1, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[1].set_title('Train Accuracy')\n",
    "axes[2].plot(df['epoch'], df['loss'], 'r-^', markersize=4)\n",
    "axes[2].axvline(x=EPOCHS_P1, color='gray', linestyle='--', alpha=0.5)\n",
    "axes[2].set_title('Loss')\n",
    "plt.tight_layout()\n",
    "plt.savefig('training_curves.png', dpi=150)\n",
    "plt.show()"
])

# Per-class analysis
add_markdown("## 8. Per-Class Analysis")
add_code([
    "if os.path.exists('best_final.pt'):\n",
    "    model.load_state_dict(torch.load('best_final.pt'))\n",
    "elif os.path.exists('best_p1.pt'):\n",
    "    model.load_state_dict(torch.load('best_p1.pt'))\n",
    "model.eval()\n",
    "\n",
    "all_preds, all_true = [], []\n",
    "with torch.no_grad():\n",
    "    for videos, video_ids in tqdm(test_loader, desc='Final Eval'):\n",
    "        videos = videos.to(DEVICE)\n",
    "        preds = model(videos).logits.argmax(1).cpu().tolist()\n",
    "        for vid, pred in zip(video_ids.tolist(), preds):\n",
    "            all_true.append(gt_dict[str(vid)])\n",
    "            all_preds.append(train_dataset.classes[pred])\n",
    "\n",
    "overall_acc = accuracy_score(all_true, all_preds)\n",
    "print(f'\\n{\"=\"*60}')\n",
    "print(f'OVERALL TEST ACCURACY: {overall_acc:.4f} ({overall_acc*100:.2f}%)')\n",
    "print(f'{\"=\"*60}')\n",
    "\n",
    "report = classification_report(all_true, all_preds, target_names=train_dataset.classes, digits=4, zero_division=0)\n",
    "print(report)\n",
    "\n",
    "with open('per_class_report.txt', 'w') as f:\n",
    "    f.write(f'Overall: {overall_acc:.4f}\\n\\n{report}')\n",
    "print('✓ Saved per_class_report.txt')"
])

# Save notebook
output_path = "/mnt/hdd/Learning/AIO-Projects/Project-7.1/videoMAE-GPU.ipynb"
with open(output_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"\n✅ Created: {output_path}")
print("\nGPU Notebook:")
print("  - Batch size: 20, Epochs: 10+5")
print("  - AMP (Mixed Precision)")
print("  - Offline augmentation with skip check")
print("  - Per-class analysis")

#!/usr/bin/env python3
"""
VideoMAE Ablation Study - Dual GPU Parallel Script
Run with: python scripts/ablation_dual_gpu.py

Features:
- 10 Experiments (8 original + 2 LR comparison)
- Dual GPU parallel execution using spawn
- Per-epoch test evaluation
- Training curves + Summary plots
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Must use spawn for CUDA multiprocessing
mp.set_start_method('spawn', force=True)

# ============================================================
# Configuration
# ============================================================
PATH_DATA_TRAIN = Path('/kaggle/input/action-video/data/data_train')
PATH_DATA_TEST = Path('/kaggle/input/action-video/data/test')

MODEL_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"
VIT_CKPT = "vit_small_patch16_224"

NUM_FRAMES = 16
IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 10
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1

# ============================================================
# Models
# ============================================================
def get_videomae_model(num_classes, device):
    from transformers import VideoMAEForVideoClassification
    label2id = {str(i): i for i in range(num_classes)}
    id2label = {i: str(i) for i in range(num_classes)}
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_CKPT, label2id=label2id, id2label=id2label,
        ignore_mismatched_sizes=True, num_frames=NUM_FRAMES
    )
    return model.to(device)

def get_vit_model(num_classes, device):
    import timm
    class ViTForAction(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(VIT_CKPT, pretrained=True, num_classes=0)
            self.head = nn.Linear(self.vit.num_features, num_classes)
        def forward(self, video):
            B, T, C, H, W = video.shape
            x = video.view(B * T, C, H, W)
            features = self.vit(x).view(B, T, -1).mean(dim=1)
            return self.head(features)
    return ViTForAction().to(device)

# ============================================================
# Datasets
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, root, consistent=False):
        self.root = Path(root)
        self.consistent = consistent
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            for vid in (self.root / cls).iterdir():
                if vid.is_dir():
                    self.samples.append((vid, self.class_to_idx[cls]))
        # Get normalization values
        from transformers import VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        self.mean = processor.image_mean
        self.std = processor.image_std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        files = sorted(vid_dir.glob('*.jpg'))
        indices = torch.linspace(0, len(files)-1, NUM_FRAMES).long()
        
        if self.consistent:
            frames = [Image.open(files[i]).convert('RGB') for i in indices]
            frames = [TF.resize(img, RESIZE_SIZE) for img in frames]
            i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33))
            is_flip = random.random() > 0.5
            transformed = []
            for img in frames:
                img = TF.resized_crop(img, i, j, h, w, size=(IMAGE_SIZE, IMAGE_SIZE))
                if is_flip:
                    img = TF.hflip(img)
                img = TF.normalize(TF.to_tensor(img), self.mean, self.std)
                transformed.append(img)
            return torch.stack(transformed), label
        else:
            frames = [TF.to_tensor(Image.open(files[i]).convert('RGB')) for i in indices]
            frames = torch.stack(frames)
            h, w = frames.shape[-2:]
            scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            frames = TF.resize(frames, [new_h, new_w])
            i = random.randint(0, max(0, new_h - IMAGE_SIZE))
            j = random.randint(0, max(0, new_w - IMAGE_SIZE))
            frames = TF.crop(frames, i, j, min(IMAGE_SIZE, new_h), min(IMAGE_SIZE, new_w))
            frames = TF.resize(frames, [IMAGE_SIZE, IMAGE_SIZE])
            if random.random() < 0.5:
                frames = TF.hflip(frames)
            from transformers import VideoMAEImageProcessor
            processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
            return torch.stack([TF.normalize(f, processor.image_mean, processor.image_std) for f in frames]), label

class TestDataset(Dataset):
    def __init__(self, root, multi_view=False, flip_tta=False):
        self.root = Path(root)
        self.multi_view = multi_view
        self.flip_tta = flip_tta
        self.samples = [(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()]
        self.samples.sort(key=lambda x: x[1])
        from transformers import VideoMAEImageProcessor
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        self.mean = processor.image_mean
        self.std = processor.image_std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vid_dir, vid_id = self.samples[idx]
        files = sorted(vid_dir.glob('*.jpg'))
        indices = torch.linspace(0, len(files)-1, NUM_FRAMES).long()
        frames = [Image.open(files[i]).convert('RGB') for i in indices]
        frames = [TF.resize(img, RESIZE_SIZE) for img in frames]
        
        if self.flip_tta:
            w, h = frames[0].size
            views = []
            for top, left in [((h-IMAGE_SIZE)//2, (w-IMAGE_SIZE)//2), (0, (w-IMAGE_SIZE)//2), (max(0,h-IMAGE_SIZE), (w-IMAGE_SIZE)//2)]:
                views.append(torch.stack([TF.normalize(TF.to_tensor(TF.crop(img, top, left, IMAGE_SIZE, IMAGE_SIZE)), self.mean, self.std) for img in frames]))
                views.append(torch.stack([TF.normalize(TF.to_tensor(TF.hflip(TF.crop(img, top, left, IMAGE_SIZE, IMAGE_SIZE))), self.mean, self.std) for img in frames]))
            return torch.stack(views), vid_id
        elif self.multi_view:
            w, h = frames[0].size
            views = []
            for top, left in [(0, 0), ((h-IMAGE_SIZE)//2, (w-IMAGE_SIZE)//2), (max(0,h-IMAGE_SIZE), max(0,w-IMAGE_SIZE))]:
                views.append(torch.stack([TF.normalize(TF.to_tensor(TF.crop(img, top, left, IMAGE_SIZE, IMAGE_SIZE)), self.mean, self.std) for img in frames]))
            return torch.stack(views), vid_id
        else:
            return torch.stack([TF.normalize(TF.to_tensor(TF.center_crop(img, IMAGE_SIZE)), self.mean, self.std) for img in frames]), vid_id

# ============================================================
# Mixup
# ============================================================
class MixupCollate:
    def __init__(self, num_classes, alpha=0.8):
        self.num_classes, self.alpha = num_classes, alpha
    def __call__(self, batch):
        inputs, targets = torch.utils.data.default_collate(batch)
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(inputs.size(0))
        inputs = lam * inputs + (1 - lam) * inputs[idx]
        onehot = F.one_hot(targets, self.num_classes).float()
        return inputs, lam * onehot + (1 - lam) * onehot[idx]

# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, device, use_mixup=False, label_smoothing=0.0, is_vit=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    for batch_idx, (videos, targets) in enumerate(loader):
        videos, targets = videos.to(device), targets.to(device)
        with torch.amp.autocast(device_type='cuda'):
            logits = model(videos) if is_vit else model(videos).logits
            if use_mixup:
                loss = -torch.sum(targets * F.log_softmax(logits, dim=1), dim=1).mean()
                true_labels = targets.argmax(dim=1)
            else:
                loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
                true_labels = targets
        correct += (logits.argmax(dim=1) == true_labels).sum().item()
        total += true_labels.size(0)
        total_loss += loss.item() * true_labels.size(0)
        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
    return total_loss / total, correct / total

def evaluate(model, loader, device, multi_view=False, is_vit=False):
    model.eval()
    all_preds, all_ids = [], []
    with torch.no_grad():
        for data, vid_ids in loader:
            if multi_view:
                B, V, T, C, H, W = data.shape
                data = data.view(B * V, T, C, H, W).to(device)
                logits = model(data) if is_vit else model(data).logits
                logits = logits.view(B, V, -1).mean(dim=1)
            else:
                data = data.to(device)
                logits = model(data) if is_vit else model(data).logits
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_ids.extend(vid_ids.tolist())
    return list(zip(all_ids, all_preds))

# ============================================================
# Single Experiment Runner
# ============================================================
def run_experiment(exp_config, gpu_id, result_queue):
    """Run single experiment on specified GPU."""
    from transformers import get_cosine_schedule_with_warmup
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    exp_name = exp_config['name']
    lr = exp_config.get('lr', 5e-5)
    print(f"[GPU {gpu_id}] {exp_name} | LR={lr}")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create datasets
    train_ds = VideoDataset(PATH_DATA_TRAIN, consistent=exp_config.get('consistent', False))
    test_ds = TestDataset(PATH_DATA_TEST, 
                          multi_view=exp_config.get('multi_view', False),
                          flip_tta=exp_config.get('flip_tta', False))
    
    # Create model
    num_classes = len(train_ds.classes)
    if exp_config.get('is_vit', False):
        model = get_vit_model(num_classes, device)
        is_vit = True
    else:
        model = get_videomae_model(num_classes, device)
        is_vit = False
    
    # Load test labels
    gt_df = pd.read_csv('test_labels.csv')
    gt_labels = dict(zip(gt_df['id'].astype(str), gt_df['class']))
    class_names = train_ds.classes
    
    def calc_acc(preds):
        y_pred, y_true = [], []
        for vid_id, pred_idx in preds:
            if str(vid_id) in gt_labels:
                y_pred.append(class_names[pred_idx])
                y_true.append(gt_labels[str(vid_id)])
        return accuracy_score(y_true, y_pred)
    
    # DataLoaders
    collate_fn = MixupCollate(num_classes) if exp_config.get('use_mixup') else None
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4 if exp_config.get('multi_view') or exp_config.get('flip_tta') else BATCH_SIZE,
                             shuffle=False, num_workers=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()
    num_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * WARMUP_RATIO), num_steps)
    
    # Training
    history = {'epoch': [], 'loss_train': [], 'acc_train': [], 'acc_test': []}
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        loss, acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                                use_mixup=exp_config.get('use_mixup', False),
                                label_smoothing=exp_config.get('label_smoothing', 0.0),
                                is_vit=is_vit)
        preds = evaluate(model, test_loader, device, 
                        multi_view=exp_config.get('multi_view') or exp_config.get('flip_tta'),
                        is_vit=is_vit)
        test_acc = calc_acc(preds)
        
        print(f"[GPU {gpu_id}] {exp_name} Ep {epoch+1}/{EPOCHS}: Loss={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
        
        history['epoch'].append(epoch + 1)
        history['loss_train'].append(loss)
        history['acc_train'].append(acc)
        history['acc_test'].append(test_acc)
        
        if acc > best_acc:
            best_acc = acc
    
    # 2-Stage Phase 2
    if exp_config.get('two_stage', False):
        print(f"[GPU {gpu_id}] {exp_name} Phase 2...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=WEIGHT_DECAY)
        p2_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(p2_loader) * 3 // GRAD_ACCUM_STEPS)
        for epoch in range(3):
            loss, acc = train_epoch(model, p2_loader, optimizer, scheduler, scaler, device, 
                                    label_smoothing=0.1, is_vit=is_vit)
            preds = evaluate(model, test_loader, device, 
                            multi_view=exp_config.get('multi_view') or exp_config.get('flip_tta'),
                            is_vit=is_vit)
            test_acc = calc_acc(preds)
            print(f"[GPU {gpu_id}] {exp_name} P2 Ep {epoch+1}/3: Loss={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
            history['epoch'].append(EPOCHS + epoch + 1)
            history['loss_train'].append(loss)
            history['acc_train'].append(acc)
            history['acc_test'].append(test_acc)
    
    final_test_acc = history['acc_test'][-1]
    print(f"[GPU {gpu_id}] {exp_name} >>> FINAL: {final_test_acc:.4f}")
    
    result_queue.put({
        'exp': exp_name,
        'train_acc': best_acc,
        'test_acc': final_test_acc,
        'gpu': gpu_id,
        'lr': lr,
        'history': history
    })
    
    del model
    torch.cuda.empty_cache()

# ============================================================
# Parallel Runner
# ============================================================
def run_pair(exp0, exp1, results):
    """Run 2 experiments in parallel on 2 GPUs."""
    q = mp.Queue()
    
    p0 = mp.Process(target=run_experiment, args=(exp0, 0, q))
    p1 = mp.Process(target=run_experiment, args=(exp1, 1, q))
    
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    
    while not q.empty():
        results.append(q.get())

# ============================================================
# Main
# ============================================================
def main():
    print("="*70)
    print("VideoMAE Ablation Study - Dual GPU (10 Experiments)")
    print("="*70)
    
    num_gpus = torch.cuda.device_count()
    print(f"GPUs: {num_gpus}")
    
    # Download test labels
    if not Path('test_labels.csv').exists():
        os.system('gdown "1Xv2CWOqdBj3kt0rkNJKRsodSIEd3-wX_" -O test_labels.csv -q')
    
    # 10 Experiments (including LR comparison)
    EXPERIMENTS = [
        # Pair 1: ViT vs VideoMAE baseline
        {'name': 'Exp0_ViT_Baseline', 'is_vit': True, 'lr': 5e-5},
        {'name': 'Exp1_VideoMAE_Paper', 'lr': 5e-5},
        
        # Pair 2: LR Comparison
        {'name': 'Exp1b_VideoMAE_LR_High', 'lr': 1.25e-4},  # Compare with 5e-5
        {'name': 'Exp2_MultiSegment_TTA', 'multi_view': True, 'lr': 5e-5},
        
        # Pair 3: Consistent + Mixup
        {'name': 'Exp3_Consistent', 'consistent': True, 'lr': 5e-5},
        {'name': 'Exp4_Mixup', 'consistent': True, 'use_mixup': True, 'lr': 5e-5},
        
        # Pair 4: Label Smoothing + 2-Stage
        {'name': 'Exp5_LabelSmooth', 'consistent': True, 'label_smoothing': 0.1, 'lr': 5e-5},
        {'name': 'Exp6_2Stage', 'consistent': True, 'use_mixup': True, 'two_stage': True, 'lr': 5e-5},
        
        # Pair 5: Flip TTA + Full
        {'name': 'Exp7_FlipTTA', 'consistent': True, 'use_mixup': True, 'two_stage': True, 'flip_tta': True, 'lr': 5e-5},
        {'name': 'Exp8_Full_LR_Compare', 'consistent': True, 'use_mixup': True, 'two_stage': True, 'flip_tta': True, 'lr': 1.25e-4},
    ]
    
    RESULTS = []
    
    if num_gpus >= 2:
        for i in range(0, len(EXPERIMENTS), 2):
            print(f"\n{'='*60}")
            print(f"Pair: {EXPERIMENTS[i]['name']} (GPU 0) | {EXPERIMENTS[i+1]['name']} (GPU 1)")
            print('='*60)
            run_pair(EXPERIMENTS[i], EXPERIMENTS[i+1], RESULTS)
    else:
        print("Running sequentially (1 GPU)...")
        q = mp.Queue()
        for exp in EXPERIMENTS:
            run_experiment(exp, 0, q)
            RESULTS.append(q.get())
    
    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    df = pd.DataFrame(RESULTS)
    df = df.sort_values('exp').reset_index(drop=True)
    print(df[['exp', 'lr', 'train_acc', 'test_acc', 'gpu']].to_string(index=False))
    
    # LR Comparison
    print("\n--- LR Comparison ---")
    lr_5e5 = df[df['exp'] == 'Exp1_VideoMAE_Paper']['test_acc'].values[0]
    lr_high = df[df['exp'] == 'Exp1b_VideoMAE_LR_High']['test_acc'].values[0]
    print(f"LR 5e-5:    {lr_5e5:.4f}")
    print(f"LR 1.25e-4: {lr_high:.4f}")
    print(f"Difference: {(lr_5e5 - lr_high)*100:+.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#e74c3c' if '1.25e-4' in str(r['lr']) or 'High' in r['exp'] else '#3498db' for r in RESULTS]
    bars = ax.bar(range(len(RESULTS)), [r['test_acc']*100 for r in RESULTS], color=colors)
    ax.set_xticks(range(len(RESULTS)))
    ax.set_xticklabels([r['exp'].replace('Exp', '').replace('_', '\n') for r in RESULTS], rotation=0, fontsize=8)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Ablation Study Results (Blue=LR 5e-5, Red=LR 1.25e-4)', fontsize=12, fontweight='bold')
    for bar, r in zip(bars, RESULTS):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{r['test_acc']*100:.1f}%", 
                ha='center', fontsize=8, fontweight='bold')
    ax.set_ylim([50, 100])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=150)
    print("\nSaved: ablation_results.png")
    plt.show()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Worker GPU 0 - Runs experiments on GPU 0
Saves results to results_gpu0.csv
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from sklearn.metrics import accuracy_score
import timm
import sys

# ============================================================
# Configuration
# ============================================================
PATH_DATA_TRAIN = Path('/kaggle/input/action-video/data/data_train')
PATH_DATA_TEST = Path('/kaggle/input/action-video/data/test')
MODEL_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"

NUM_FRAMES = 16
IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 10
LR = 5e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1

device = torch.device('cuda:0')
processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
MEAN, STD = processor.image_mean, processor.image_std

# ============================================================
# Models
# ============================================================
class ViTForAction(nn.Module):
    def __init__(self, num_classes=51):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        self.head = nn.Linear(self.vit.num_features, num_classes)
    def forward(self, video):
        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)
        return self.head(self.vit(x).view(B, T, -1).mean(dim=1))

# ============================================================
# Datasets
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, root, consistent=False):
        self.root = Path(root)
        self.consistent = consistent
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(vid, self.class_to_idx[cls]) for cls in self.classes for vid in (self.root / cls).iterdir() if vid.is_dir()]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vid_dir, label = self.samples[idx]
        files = sorted(vid_dir.glob('*.jpg'))
        indices = torch.linspace(0, len(files)-1, NUM_FRAMES).long()
        if self.consistent:
            frames = [TF.resize(Image.open(files[i]).convert('RGB'), RESIZE_SIZE) for i in indices]
            i, j, h, w = T.RandomResizedCrop.get_params(frames[0], scale=(0.8, 1.0), ratio=(0.75, 1.33))
            flip = random.random() > 0.5
            return torch.stack([TF.normalize(TF.to_tensor(TF.hflip(TF.resized_crop(f, i, j, h, w, (IMAGE_SIZE, IMAGE_SIZE))) if flip else TF.resized_crop(f, i, j, h, w, (IMAGE_SIZE, IMAGE_SIZE))), MEAN, STD) for f in frames]), label
        else:
            frames = torch.stack([TF.to_tensor(Image.open(files[i]).convert('RGB')) for i in indices])
            scale = random.uniform(0.8, 1.0)
            h, w = frames.shape[-2:]
            frames = TF.resize(frames, [int(h*scale), int(w*scale)])
            i, j = random.randint(0, max(0, frames.shape[-2]-IMAGE_SIZE)), random.randint(0, max(0, frames.shape[-1]-IMAGE_SIZE))
            frames = TF.resize(TF.crop(frames, i, j, min(IMAGE_SIZE, frames.shape[-2]), min(IMAGE_SIZE, frames.shape[-1])), [IMAGE_SIZE, IMAGE_SIZE])
            if random.random() < 0.5: frames = TF.hflip(frames)
            return torch.stack([TF.normalize(f, MEAN, STD) for f in frames]), label

class TestDataset(Dataset):
    def __init__(self, root, flip_tta=False):
        self.root = Path(root)
        self.flip_tta = flip_tta
        self.samples = sorted([(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()], key=lambda x: x[1])
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        vid_dir, vid_id = self.samples[idx]
        files = sorted(vid_dir.glob('*.jpg'))
        indices = torch.linspace(0, len(files)-1, NUM_FRAMES).long()
        frames = [TF.resize(Image.open(files[i]).convert('RGB'), RESIZE_SIZE) for i in indices]
        if self.flip_tta:
            w, h = frames[0].size
            views = []
            for top, left in [((h-IMAGE_SIZE)//2, (w-IMAGE_SIZE)//2), (0, (w-IMAGE_SIZE)//2), (max(0,h-IMAGE_SIZE), (w-IMAGE_SIZE)//2)]:
                views.append(torch.stack([TF.normalize(TF.to_tensor(TF.crop(f, top, left, IMAGE_SIZE, IMAGE_SIZE)), MEAN, STD) for f in frames]))
                views.append(torch.stack([TF.normalize(TF.to_tensor(TF.hflip(TF.crop(f, top, left, IMAGE_SIZE, IMAGE_SIZE))), MEAN, STD) for f in frames]))
            return torch.stack(views), vid_id
        return torch.stack([TF.normalize(TF.to_tensor(TF.center_crop(f, IMAGE_SIZE)), MEAN, STD) for f in frames]), vid_id

class MixupCollate:
    def __init__(self, nc, alpha=0.8): self.nc, self.alpha = nc, alpha
    def __call__(self, batch):
        x, y = torch.utils.data.default_collate(batch)
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(x.size(0))
        return lam * x + (1-lam) * x[idx], lam * F.one_hot(y, self.nc).float() + (1-lam) * F.one_hot(y[idx], self.nc).float()

# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, opt, sch, scaler, mixup=False, ls=0.0, vit=False):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(x) if vit else model(x).logits
            if mixup:
                loss = -torch.sum(y * F.log_softmax(logits, 1), 1).mean()
                labels = y.argmax(1)
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=ls)
                labels = y
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sch.step()
    return loss_sum / total, correct / total

def evaluate(model, loader, multi=False, vit=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, ids in loader:
            if multi:
                B, V, T, C, H, W = x.shape
                x = x.view(B*V, T, C, H, W).to(device)
                logits = (model(x) if vit else model(x).logits).view(B, V, -1).mean(1)
            else:
                logits = model(x.to(device)) if vit else model(x.to(device)).logits
            preds.extend(zip(ids.tolist(), logits.argmax(1).cpu().tolist()))
    return preds

# ============================================================
# Experiments for GPU 0
# ============================================================
EXPERIMENTS = [
    {'name': 'Exp0_ViT_Baseline', 'vit': True},
    {'name': 'Exp2_Consistent', 'consistent': True},
    {'name': 'Exp4_Mixup', 'consistent': True, 'mixup': True},
    {'name': 'Exp6_2Stage', 'consistent': True, 'mixup': True, 'two_stage': True},
    {'name': 'Exp8_LR_High', 'consistent': True, 'mixup': True, 'two_stage': True, 'flip_tta': True, 'lr': 1.25e-4},
]

def run_experiments():
    # Load test labels
    gt_df = pd.read_csv('test_labels.csv')
    gt_labels = dict(zip(gt_df['id'].astype(str), gt_df['class']))
    
    results = []
    
    for exp in EXPERIMENTS:
        print(f"\n{'='*50}\n[GPU 0] {exp['name']}\n{'='*50}")
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        
        lr = exp.get('lr', LR)
        train_ds = VideoDataset(PATH_DATA_TRAIN, consistent=exp.get('consistent', False))
        test_ds = TestDataset(PATH_DATA_TEST, flip_tta=exp.get('flip_tta', False))
        class_names = train_ds.classes
        
        # Model
        if exp.get('vit'):
            model = ViTForAction(len(class_names)).to(device)
        else:
            model = VideoMAEForVideoClassification.from_pretrained(
                MODEL_CKPT, num_labels=len(class_names), ignore_mismatched_sizes=True, num_frames=NUM_FRAMES
            ).to(device)
        
        collate = MixupCollate(len(class_names)) if exp.get('mixup') else None
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate)
        test_loader = DataLoader(test_ds, 4 if exp.get('flip_tta') else BATCH_SIZE, num_workers=2)
        
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler()
        sch = get_cosine_schedule_with_warmup(opt, int(len(train_loader)*EPOCHS*WARMUP_RATIO/GRAD_ACCUM_STEPS), len(train_loader)*EPOCHS//GRAD_ACCUM_STEPS)
        
        history = []
        for epoch in range(EPOCHS):
            loss, acc = train_epoch(model, train_loader, opt, sch, scaler, exp.get('mixup'), 0.0, exp.get('vit'))
            preds = evaluate(model, test_loader, exp.get('flip_tta'), exp.get('vit'))
            test_acc = accuracy_score([gt_labels[str(i)] for i, _ in preds], [class_names[p] for _, p in preds])
            print(f"  Ep {epoch+1}/{EPOCHS}: Loss={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
            history.append({'epoch': epoch+1, 'loss': loss, 'acc_train': acc, 'acc_test': test_acc})
        
        if exp.get('two_stage'):
            print("  Phase 2...")
            opt = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=WEIGHT_DECAY)
            p2_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
            sch = get_cosine_schedule_with_warmup(opt, 0, len(p2_loader)*3//GRAD_ACCUM_STEPS)
            for epoch in range(3):
                loss, acc = train_epoch(model, p2_loader, opt, sch, scaler, False, 0.1, exp.get('vit'))
                preds = evaluate(model, test_loader, exp.get('flip_tta'), exp.get('vit'))
                test_acc = accuracy_score([gt_labels[str(i)] for i, _ in preds], [class_names[p] for _, p in preds])
                print(f"  P2 Ep {epoch+1}/3: Loss={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
                history.append({'epoch': EPOCHS+epoch+1, 'loss': loss, 'acc_train': acc, 'acc_test': test_acc})
        
        final_acc = history[-1]['acc_test']
        print(f"  >>> FINAL: {final_acc:.4f}")
        results.append({'exp': exp['name'], 'test_acc': final_acc, 'lr': lr, 'gpu': 0})
        
        del model
        torch.cuda.empty_cache()
    
    pd.DataFrame(results).to_csv('results_gpu0.csv', index=False)
    print("\nSaved: results_gpu0.csv")

if __name__ == '__main__':
    run_experiments()

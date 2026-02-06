#!/usr/bin/env python3
"""
Generate incremental-ablation.ipynb - Ablation study with cumulative technique integration.

Strategy: Add techniques one-by-one and measure cumulative improvement.
- Exp1: VideoMAE Baseline
- Exp2: + Consistent Spatial Aug
- Exp3: + Mixup
- Exp4: + Label Smoothing (2-Stage)
- Exp5: + 6-View TTA (Inference only)

Epochs: 10 per experiment (Phase 2 uses 3 epochs if 2-Stage is enabled)
"""

import json


def create_notebook():
    """Create the incremental ablation notebook."""
    
    cells = []
    
    # ==================== MARKDOWN: Title ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Incremental Ablation Study - VideoMAE\n",
            "\n",
            "**Chiến lược**: Tích hợp từng kỹ thuật một cách tuần tự để đo lường cải thiện tích lũy.\n",
            "\n",
            "| Exp | Name | Techniques | Expected Acc |\n",
            "|-----|------|------------|--------------|\n",
            "| 1 | Baseline | VideoMAE (LR=5e-5) | ~83.92% |\n",
            "| 2 | +ConsistentAug | + Consistent Spatial Augmentation | ~84.31% |\n",
            "| 3 | +Mixup | + Mixup (α=0.8) | ~82.55%* |\n",
            "| 4 | +2Stage | + 2-Stage (Mixup→Label Smoothing) | ~84.71% |\n",
            "| 5 | +TTA | + 6-View TTA at Inference | **~85.10%** |\n",
            "\n",
            "*Mixup đơn lẻ giảm accuracy ngắn hạn nhưng khi kết hợp với 2-Stage sẽ hiệu quả hơn.\n",
            "\n",
            "**Epochs**: 10 (Phase 1: 7, Phase 2: 3 nếu 2-Stage)\n"
        ]
    })
    
    # ==================== CODE: Configuration ====================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# =================== CONFIGURATION ===================\n",
            "# ===== EXPERIMENT SELECTION =====\n",
            "RUN_EXP = 4                 # Which experiment to run: 1, 2, 3, or 4 (4 includes TTA)\n",
            "                            # Run separately on different Kaggle sessions\n",
            "\n",
            "# ===== QUICK TEST MODE =====\n",
            "QUICK_TEST = False          # True = test pipeline with 5 batches per phase\n",
            "QUICK_TEST_BATCHES = 5      # Number of batches when QUICK_TEST=True\n",
            "\n",
            "# Model Config\n",
            "MODEL_CKPT = 'MCG-NJU/videomae-base-finetuned-kinetics'\n",
            "NUM_FRAMES = 16\n",
            "IMG_SIZE = 224\n",
            "RESIZE_SIZE = 256\n",
            "\n",
            "# Training Config\n",
            "EPOCHS_TOTAL = 1 if QUICK_TEST else 10\n",
            "EPOCHS_P1 = 1 if QUICK_TEST else 7      # Phase 1 epochs (when 2-Stage)\n",
            "EPOCHS_P2 = 1 if QUICK_TEST else 3      # Phase 2 epochs (when 2-Stage)\n",
            "BATCH_SIZE = 8\n",
            "ACCUM_STEPS = 4\n",
            "LR_P1 = 5e-5\n",
            "LR_P2 = 1e-6\n",
            "WEIGHT_DECAY = 0.05\n",
            "WARMUP_RATIO = 0.1\n",
            "\n",
            "# Augmentation Config\n",
            "MIXUP_ALPHA = 0.8\n",
            "LABEL_SMOOTHING_EPS = 0.1\n",
            "\n",
            "# Paths (Kaggle)\n",
            "PATH_DATA_TRAIN = '/kaggle/input/action-video/data/data_train'\n",
            "PATH_DATA_TEST = '/kaggle/input/action-video/data/test'\n",
            "TEST_LABELS_URL = '1Xv2CWOqdBj3kt0rkNJKRsodSIEd3-wX_'\n",
            "\n",
            "EXP_NAMES = {1: 'Baseline', 2: '+ConsistentAug', 3: '+Mixup', 4: '+2Stage+TTA'}\n",
            "print('='*60)\n",
            "print(f'RUNNING: Exp {RUN_EXP} - {EXP_NAMES[RUN_EXP]}')\n",
            "if QUICK_TEST:\n",
            "    print(f'  [QUICK_TEST] Only {QUICK_TEST_BATCHES} batches, 1 epoch')\n",
            "print('='*60)"
        ]
    })
    
    # ==================== CODE: Imports ====================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n",
            "import gc\n",
            "import random\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "from pathlib import Path\n",
            "from collections import Counter\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "from tqdm.auto import tqdm\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import torchvision.transforms as T\n",
            "import torchvision.transforms.functional as TF\n",
            "\n",
            "from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor\n",
            "from transformers import get_cosine_schedule_with_warmup\n",
            "from sklearn.metrics import accuracy_score\n",
            "\n",
            "# Seed\n",
            "def seed_everything(seed=42):\n",
            "    random.seed(seed)\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    torch.cuda.manual_seed_all(seed)\n",
            "\n",
            "seed_everything(42)\n",
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "print(f'Device: {DEVICE}')\n",
            "\n",
            "MEAN = [0.485, 0.456, 0.406]\n",
            "STD = [0.229, 0.224, 0.225]"
        ]
    })
    
    # ==================== CODE: Dataset Classes ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Dataset Classes"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class VideoDataset(Dataset):\n",
            "    \"\"\"Training dataset with toggle for consistent spatial aug.\"\"\"\n",
            "    def __init__(self, root, num_frames=16, consistent_aug=False):\n",
            "        self.root = Path(root)\n",
            "        self.num_frames = num_frames\n",
            "        self.consistent_aug = consistent_aug\n",
            "        \n",
            "        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])\n",
            "        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}\n",
            "        \n",
            "        self.samples = []\n",
            "        for cls in self.classes:\n",
            "            for vid in (self.root / cls).iterdir():\n",
            "                if vid.is_dir():\n",
            "                    self.samples.append((vid, self.class_to_idx[cls]))\n",
            "        print(f'Loaded {len(self.samples)} videos, {len(self.classes)} classes')\n",
            "    \n",
            "    def __len__(self):\n",
            "        return len(self.samples)\n",
            "    \n",
            "    def __getitem__(self, idx):\n",
            "        vid_dir, label = self.samples[idx]\n",
            "        files = sorted(vid_dir.glob('*.jpg'))\n",
            "        indices = torch.linspace(0, len(files)-1, self.num_frames).long()\n",
            "        \n",
            "        frames = [Image.open(files[i]).convert('RGB') for i in indices]\n",
            "        frames = [TF.resize(img, RESIZE_SIZE) for img in frames]\n",
            "        \n",
            "        if self.consistent_aug:\n",
            "            # Same crop/flip for all frames\n",
            "            i, j, h, w = T.RandomResizedCrop.get_params(frames[0], (0.8, 1.0), (0.75, 1.33))\n",
            "            do_flip = random.random() > 0.5\n",
            "            processed = []\n",
            "            for img in frames:\n",
            "                img = TF.resized_crop(img, i, j, h, w, (IMG_SIZE, IMG_SIZE))\n",
            "                if do_flip:\n",
            "                    img = TF.hflip(img)\n",
            "                img = TF.normalize(TF.to_tensor(img), MEAN, STD)\n",
            "                processed.append(img)\n",
            "        else:\n",
            "            # Simple center crop (no aug)\n",
            "            processed = [TF.normalize(TF.to_tensor(TF.center_crop(img, IMG_SIZE)), MEAN, STD) \n",
            "                         for img in frames]\n",
            "        \n",
            "        return torch.stack(processed), label\n",
            "\n",
            "\n",
            "class TestDatasetSingle(Dataset):\n",
            "    \"\"\"Test with single center crop.\"\"\"\n",
            "    def __init__(self, root, num_frames=16):\n",
            "        self.root = Path(root)\n",
            "        self.num_frames = num_frames\n",
            "        self.samples = [(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()]\n",
            "        self.samples.sort(key=lambda x: x[1])\n",
            "    \n",
            "    def __len__(self):\n",
            "        return len(self.samples)\n",
            "    \n",
            "    def __getitem__(self, idx):\n",
            "        vid_dir, vid_id = self.samples[idx]\n",
            "        files = sorted(vid_dir.glob('*.jpg'))\n",
            "        indices = torch.linspace(0, len(files)-1, self.num_frames).long()\n",
            "        frames = []\n",
            "        for i in indices:\n",
            "            img = Image.open(files[i]).convert('RGB')\n",
            "            img = TF.resize(img, RESIZE_SIZE)\n",
            "            img = TF.center_crop(img, IMG_SIZE)\n",
            "            img = TF.normalize(TF.to_tensor(img), MEAN, STD)\n",
            "            frames.append(img)\n",
            "        return torch.stack(frames), vid_id\n",
            "\n",
            "\n",
            "class TestDatasetTTA(Dataset):\n",
            "    \"\"\"6-view TTA: 3 spatial crops × 2 flip states.\"\"\"\n",
            "    def __init__(self, root, num_frames=16):\n",
            "        self.root = Path(root)\n",
            "        self.num_frames = num_frames\n",
            "        self.samples = [(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()]\n",
            "        self.samples.sort(key=lambda x: x[1])\n",
            "    \n",
            "    def __len__(self):\n",
            "        return len(self.samples)\n",
            "    \n",
            "    def __getitem__(self, idx):\n",
            "        vid_dir, vid_id = self.samples[idx]\n",
            "        files = sorted(vid_dir.glob('*.jpg'))\n",
            "        indices = torch.linspace(0, len(files)-1, self.num_frames).long()\n",
            "        \n",
            "        frames = [Image.open(files[i]).convert('RGB') for i in indices]\n",
            "        frames = [TF.resize(img, RESIZE_SIZE) for img in frames]\n",
            "        \n",
            "        w, h = frames[0].size\n",
            "        views = []\n",
            "        \n",
            "        # 3 spatial crops\n",
            "        crop_positions = [\n",
            "            ((h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2),  # center\n",
            "            (0, 0),  # top-left\n",
            "            (max(0, h - IMG_SIZE), max(0, w - IMG_SIZE))  # bottom-right\n",
            "        ]\n",
            "        \n",
            "        for top, left in crop_positions:\n",
            "            view_frames = []\n",
            "            for img in frames:\n",
            "                cropped = TF.crop(img, top, left, IMG_SIZE, IMG_SIZE)\n",
            "                view_frames.append(TF.normalize(TF.to_tensor(cropped), MEAN, STD))\n",
            "            views.append(torch.stack(view_frames))\n",
            "            \n",
            "            # Flipped version\n",
            "            view_frames_flip = []\n",
            "            for img in frames:\n",
            "                cropped = TF.crop(img, top, left, IMG_SIZE, IMG_SIZE)\n",
            "                cropped = TF.hflip(cropped)\n",
            "                view_frames_flip.append(TF.normalize(TF.to_tensor(cropped), MEAN, STD))\n",
            "            views.append(torch.stack(view_frames_flip))\n",
            "        \n",
            "        return torch.stack(views), vid_id  # [6, T, C, H, W]\n",
            "\n",
            "print('Dataset classes defined.')"
        ]
    })
    
    # ==================== CODE: Mixup Collate ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Training Utilities"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class MixupCollate:\n",
            "    \"\"\"Collate function with Mixup augmentation.\"\"\"\n",
            "    def __init__(self, num_classes, alpha=0.8):\n",
            "        self.num_classes = num_classes\n",
            "        self.alpha = alpha\n",
            "\n",
            "    def __call__(self, batch):\n",
            "        inputs, targets = torch.utils.data.default_collate(batch)\n",
            "        batch_size = inputs.size(0)\n",
            "        index = torch.randperm(batch_size)\n",
            "        lam = np.random.beta(self.alpha, self.alpha)\n",
            "        inputs = lam * inputs + (1 - lam) * inputs[index, :]\n",
            "        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()\n",
            "        targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index, :]\n",
            "        return inputs, targets\n",
            "\n",
            "\n",
            "def train_epoch(model, loader, optimizer, scheduler, scaler, \n",
            "                use_mixup=False, label_smoothing=0.0, max_batches=None):\n",
            "    \"\"\"Train one epoch. If max_batches is set, stop early for quick testing.\"\"\"\n",
            "    model.train()\n",
            "    total_loss, total_correct, total_samples = 0.0, 0, 0\n",
            "    pbar = tqdm(loader, desc='Training', leave=False)\n",
            "    optimizer.zero_grad()\n",
            "    \n",
            "    for step, (inputs, targets) in enumerate(pbar):\n",
            "        if max_batches is not None and step >= max_batches:\n",
            "            break\n",
            "        \n",
            "        inputs = inputs.to(DEVICE)\n",
            "        \n",
            "        with torch.amp.autocast('cuda'):\n",
            "            output = model(inputs)\n",
            "            logits = output.logits if hasattr(output, 'logits') else output\n",
            "            \n",
            "            if use_mixup:\n",
            "                targets = targets.to(DEVICE)\n",
            "                log_probs = F.log_softmax(logits, dim=1)\n",
            "                loss = -torch.sum(targets * log_probs, dim=1).mean()\n",
            "                true_labels = targets.argmax(dim=1)\n",
            "            else:\n",
            "                targets = targets.to(DEVICE)\n",
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
            "            if scheduler:\n",
            "                scheduler.step()\n",
            "        \n",
            "        total_loss += loss.item()\n",
            "        pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}', \n",
            "                          'acc': f'{total_correct/max(total_samples,1):.4f}'})\n",
            "    \n",
            "    return total_loss / max(step+1, 1), total_correct / max(total_samples, 1)\n",
            "\n",
            "\n",
            "@torch.no_grad()\n",
            "def evaluate(model, loader, classes, gt_dict, use_tta=False, max_batches=None):\n",
            "    \"\"\"Evaluate model on test set. If max_batches is set, stop early for quick testing.\"\"\"\n",
            "    model.eval()\n",
            "    predictions = []\n",
            "    \n",
            "    for step, (videos, video_ids) in enumerate(tqdm(loader, desc='Evaluating', leave=False)):\n",
            "        if max_batches is not None and step >= max_batches:\n",
            "            break\n",
            "        \n",
            "        if use_tta:\n",
            "            B, V, T, C, H, W = videos.shape\n",
            "            videos = videos.view(B * V, T, C, H, W).to(DEVICE)\n",
            "            output = model(videos)\n",
            "            logits = output.logits if hasattr(output, 'logits') else output\n",
            "            logits = logits.view(B, V, -1).mean(dim=1)\n",
            "            preds = logits.argmax(1).cpu().tolist()\n",
            "        else:\n",
            "            videos = videos.to(DEVICE)\n",
            "            output = model(videos)\n",
            "            logits = output.logits if hasattr(output, 'logits') else output\n",
            "            preds = logits.argmax(1).cpu().tolist()\n",
            "        \n",
            "        predictions.extend(zip(video_ids.tolist(), preds))\n",
            "    \n",
            "    if max_batches is not None:\n",
            "        return 0.0  # Skip accuracy calc in quick test mode\n",
            "    \n",
            "    y_true = [gt_dict[str(vid)] for vid, _ in predictions]\n",
            "    y_pred = [classes[p] for _, p in predictions]\n",
            "    return accuracy_score(y_true, y_pred)\n",
            "\n",
            "print('Training utilities defined.')"
        ]
    })
    
    # ==================== CODE: Load Data & Labels ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Load Data"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Download test labels\n",
            "!gdown \"{TEST_LABELS_URL}\" -O test_labels.csv -q\n",
            "gt_df = pd.read_csv('test_labels.csv')\n",
            "gt_dict = dict(zip(gt_df['id'].astype(str), gt_df['class']))\n",
            "print(f'Loaded test labels: {len(gt_dict)} samples')\n",
            "\n",
            "# Results storage\n",
            "all_results = []"
        ]
    })
    
    # ==================== EXPERIMENT 1: Baseline ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Exp 1: VideoMAE Baseline\n",
            "- **Techniques**: None (just fine-tune with LR=5e-5)\n",
            "- **Expected**: ~83.92%"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if RUN_EXP == 1:\n",
            "    print('='*60)\n",
            "    print('EXP 1: BASELINE (No augmentation)')\n",
            "    print('='*60)\n",
            "\n",
            "    # Dataset without consistent aug\n",
            "    train_ds = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, consistent_aug=False)\n",
            "    test_ds = TestDatasetSingle(PATH_DATA_TEST, NUM_FRAMES)\n",
            "\n",
            "    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)\n",
            "    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)\n",
            "\n",
            "    # Model\n",
            "    model = VideoMAEForVideoClassification.from_pretrained(\n",
            "        MODEL_CKPT,\n",
            "        num_labels=len(train_ds.classes),\n",
            "        ignore_mismatched_sizes=True,\n",
            "        num_frames=NUM_FRAMES\n",
            "    ).to(DEVICE)\n",
            "\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
            "    total_steps = len(train_loader) * EPOCHS_TOTAL // ACCUM_STEPS\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
            "    scaler = torch.amp.GradScaler()\n",
            "\n",
            "    best_acc = 0.0\n",
            "    max_batches = QUICK_TEST_BATCHES if QUICK_TEST else None\n",
            "    for epoch in range(1, EPOCHS_TOTAL + 1):\n",
            "        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, max_batches=max_batches)\n",
            "        test_acc = evaluate(model, test_loader, train_ds.classes, gt_dict, max_batches=max_batches)\n",
            "        \n",
            "        status = ''\n",
            "        if test_acc > best_acc:\n",
            "            best_acc = test_acc\n",
            "            torch.save(model.state_dict(), 'exp1_best.pt')\n",
            "            status = '>>> BEST'\n",
            "        \n",
            "        print(f'Ep {epoch}/{EPOCHS_TOTAL}: Loss={loss:.4f} TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f} {status}')\n",
            "\n",
            "    print(f'\\nExp1 Best: {best_acc:.4f}')\n",
            "    all_results.append({'exp': 1, 'name': 'Baseline', 'test_acc': best_acc})\n",
            "    del model, optimizer, scheduler\n",
            "    torch.cuda.empty_cache()\n",
            "    gc.collect()\n",
            "else:\n",
            "    print('Skipping Exp1 (RUN_EXP != 1)')"
        ]
    })
    
    # ==================== EXPERIMENT 2: + Consistent Aug ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Exp 2: + Consistent Spatial Augmentation\n",
            "- **New**: Same crop/flip for all 16 frames\n",
            "- **Expected**: +0.39% (~84.31%)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "if RUN_EXP == 2:\n",
            "    print('='*60)\n",
            "    print('EXP 2: + CONSISTENT SPATIAL AUGMENTATION')\n",
            "    print('='*60)\n",
            "\n",
            "    train_ds = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, consistent_aug=True)\n",
            "    test_ds = TestDatasetSingle(PATH_DATA_TEST, NUM_FRAMES)\n",
            "    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)\n",
            "    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)\n",
            "\n",
            "    model = VideoMAEForVideoClassification.from_pretrained(\n",
            "        MODEL_CKPT, num_labels=len(train_ds.classes),\n",
            "        ignore_mismatched_sizes=True, num_frames=NUM_FRAMES\n",
            "    ).to(DEVICE)\n",
            "\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
            "    total_steps = len(train_loader) * EPOCHS_TOTAL // ACCUM_STEPS\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
            "    scaler = torch.amp.GradScaler()\n",
            "\n",
            "    best_acc = 0.0\n",
            "    max_batches = QUICK_TEST_BATCHES if QUICK_TEST else None\n",
            "    for epoch in range(1, EPOCHS_TOTAL + 1):\n",
            "        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, max_batches=max_batches)\n",
            "        test_acc = evaluate(model, test_loader, train_ds.classes, gt_dict, max_batches=max_batches)\n",
            "        status = '>>> BEST' if test_acc > best_acc else ''\n",
            "        if test_acc > best_acc:\n",
            "            best_acc = test_acc\n",
            "            torch.save(model.state_dict(), 'exp2_best.pt')\n",
            "        print(f'Ep {epoch}/{EPOCHS_TOTAL}: Loss={loss:.4f} TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f} {status}')\n",
            "\n",
            "    print(f'Exp2 Best: {best_acc:.4f}')\n",
            "    all_results.append({'exp': 2, 'name': '+ConsistentAug', 'test_acc': best_acc})\n",
            "    del model, optimizer, scheduler\n",
            "    torch.cuda.empty_cache(); gc.collect()\n",
            "else:\n",
            "    print('Skipping Exp2')"
        ]
    })
    
    # ==================== EXPERIMENT 3: + Mixup ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Exp 3: + Mixup\n",
            "- **New**: Mixup (α=0.8) on top of Consistent Aug\n",
            "- **Note**: Mixup alone may decrease accuracy but reduces overfitting gap\n",
            "- **Expected**: ~82.55% (trade-off for better generalization)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if RUN_EXP == 3:\n",
            "    print('='*60)\n",
            "    print('EXP 3: + MIXUP')\n",
            "    print('='*60)\n",
            "\n",
            "    train_ds = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, consistent_aug=True)\n",
            "    test_ds = TestDatasetSingle(PATH_DATA_TEST, NUM_FRAMES)\n",
            "    mixup_collate = MixupCollate(len(train_ds.classes), MIXUP_ALPHA)\n",
            "    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=mixup_collate, drop_last=True, persistent_workers=True)\n",
            "    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)\n",
            "\n",
            "    model = VideoMAEForVideoClassification.from_pretrained(\n",
            "        MODEL_CKPT, num_labels=len(train_ds.classes),\n",
            "        ignore_mismatched_sizes=True, num_frames=NUM_FRAMES\n",
            "    ).to(DEVICE)\n",
            "\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
            "    total_steps = len(train_loader) * EPOCHS_TOTAL // ACCUM_STEPS\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
            "    scaler = torch.amp.GradScaler()\n",
            "\n",
            "    best_acc = 0.0\n",
            "    max_batches = QUICK_TEST_BATCHES if QUICK_TEST else None\n",
            "    for epoch in range(1, EPOCHS_TOTAL + 1):\n",
            "        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, use_mixup=True, max_batches=max_batches)\n",
            "        test_acc = evaluate(model, test_loader, train_ds.classes, gt_dict, max_batches=max_batches)\n",
            "        status = '>>> BEST' if test_acc > best_acc else ''\n",
            "        if test_acc > best_acc:\n",
            "            best_acc = test_acc\n",
            "            torch.save(model.state_dict(), 'exp3_best.pt')\n",
            "        print(f'Ep {epoch}/{EPOCHS_TOTAL}: Loss={loss:.4f} TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f} {status}')\n",
            "\n",
            "    print(f'Exp3 Best: {best_acc:.4f}')\n",
            "    all_results.append({'exp': 3, 'name': '+Mixup', 'test_acc': best_acc})\n",
            "    del model, optimizer, scheduler\n",
            "    torch.cuda.empty_cache(); gc.collect()\n",
            "else:\n",
            "    print('Skipping Exp3')"
        ]
    })
    
    # ==================== EXPERIMENT 4: + 2-Stage ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Exp 4+5: 2-Stage Training + TTA\n",
            "- **Phase 1**: Mixup + LR=5e-5\n",
            "- **Phase 2**: Label Smoothing + LR=1e-6\n",
            "- **TTA**: 3 spatial crops × 2 flip states\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if RUN_EXP == 4:\n",
            "    print('='*60)\n",
            "    print('EXP 4+5: 2-STAGE TRAINING + TTA')\n",
            "    print('='*60)\n",
            "    \n",
            "    # Setup\n",
            "    train_ds = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES, consistent_aug=True)\n",
            "    test_ds = TestDatasetSingle(PATH_DATA_TEST, NUM_FRAMES)\n",
            "    mixup_collate = MixupCollate(len(train_ds.classes), MIXUP_ALPHA)\n",
            "    train_loader_p1 = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=mixup_collate, drop_last=True, persistent_workers=True)\n",
            "    train_loader_p2 = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)\n",
            "    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)\n",
            "    \n",
            "    model = VideoMAEForVideoClassification.from_pretrained(\n",
            "        MODEL_CKPT, num_labels=len(train_ds.classes),\n",
            "        ignore_mismatched_sizes=True, num_frames=NUM_FRAMES\n",
            "    ).to(DEVICE)\n",
            "    \n",
            "    best_acc = 0.0\n",
            "    max_batches = QUICK_TEST_BATCHES if QUICK_TEST else None\n",
            "    \n",
            "    # Phase 1: Mixup\n",
            "    print(f'\\\\n--- Phase 1: Mixup ({EPOCHS_P1} epochs) ---')\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n",
            "    total_steps = len(train_loader_p1) * EPOCHS_P1 // ACCUM_STEPS\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
            "    scaler = torch.amp.GradScaler()\n",
            "    \n",
            "    for epoch in range(1, EPOCHS_P1 + 1):\n",
            "        loss, train_acc = train_epoch(model, train_loader_p1, optimizer, scheduler, scaler, use_mixup=True, max_batches=max_batches)\n",
            "        test_acc = evaluate(model, test_loader, train_ds.classes, gt_dict, max_batches=max_batches)\n",
            "        if test_acc > best_acc:\n",
            "            best_acc = test_acc\n",
            "            torch.save(model.state_dict(), 'exp4_best.pt')\n",
            "        print(f'P1 Ep {epoch}/{EPOCHS_P1}: Loss={loss:.4f} Acc={test_acc:.4f}')\n",
            "    \n",
            "    if not os.path.exists('exp4_best.pt'):\n",
            "        torch.save(model.state_dict(), 'exp4_best.pt')\n",
            "    \n",
            "    # Phase 2: Label Smoothing\n",
            "    print(f'\\\\n--- Phase 2: Label Smoothing ({EPOCHS_P2} epochs) ---')\n",
            "    model.load_state_dict(torch.load('exp4_best.pt'))\n",
            "    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=WEIGHT_DECAY)\n",
            "    total_steps = len(train_loader_p2) * EPOCHS_P2 // ACCUM_STEPS\n",
            "    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)\n",
            "    scaler = torch.amp.GradScaler()\n",
            "    \n",
            "    for epoch in range(1, EPOCHS_P2 + 1):\n",
            "        loss, train_acc = train_epoch(model, train_loader_p2, optimizer, scheduler, scaler, label_smoothing=LABEL_SMOOTHING_EPS, max_batches=max_batches)\n",
            "        test_acc = evaluate(model, test_loader, train_ds.classes, gt_dict, max_batches=max_batches)\n",
            "        if test_acc > best_acc:\n",
            "            best_acc = test_acc\n",
            "            torch.save(model.state_dict(), 'exp4_best.pt')\n",
            "        print(f'P2 Ep {epoch}/{EPOCHS_P2}: Loss={loss:.4f} Acc={test_acc:.4f}')\n",
            "    \n",
            "    print(f'\\\\nExp4 (2-Stage) Best: {best_acc:.4f}')\n",
            "    all_results.append({'exp': 4, 'name': '+2Stage', 'test_acc': best_acc})\n",
            "    \n",
            "    # Exp5: TTA\n",
            "    print('\\\\n' + '='*60)\n",
            "    print('EXP 5: + 6-VIEW TTA')\n",
            "    print('='*60)\n",
            "    model.load_state_dict(torch.load('exp4_best.pt'))\n",
            "    test_ds_tta = TestDatasetTTA(PATH_DATA_TEST, NUM_FRAMES)\n",
            "    test_loader_tta = DataLoader(test_ds_tta, BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=True)\n",
            "    test_acc_tta = evaluate(model, test_loader_tta, train_ds.classes, gt_dict, use_tta=True, max_batches=max_batches)\n",
            "    print(f'\\\\nExp5 (TTA) Accuracy: {test_acc_tta:.4f}')\n",
            "    all_results.append({'exp': 5, 'name': '+TTA', 'test_acc': test_acc_tta})\n",
            "    \n",
            "    del model, optimizer, scheduler\n",
            "    torch.cuda.empty_cache()\n",
            "    gc.collect()\n",
            "else:\n",
            "    print('Skipping Exp4+5 (RUN_EXP != 4)')"
        ]
    })
    

    # Create notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


if __name__ == "__main__":
    notebook = create_notebook()
    output_path = "/mnt/hdd/Learning/AIO-Projects/Project-7.1/notebooks/incremental-ablation.ipynb"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Created {output_path}")
    print(f"  Total cells: {len(notebook['cells'])}")

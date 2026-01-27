"""
Generate analyze-checkpoint.ipynb for per-class analysis
"""
import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
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
    "# Per-Class Accuracy Analysis\n",
    "\n",
    "**Purpose**: Analyze model performance on test set\n",
    "- Download checkpoint from Google Drive\n",
    "- Evaluate per-class accuracy\n",
    "- Visualize class distribution and performance"
])

# Install & Import
add_code("!pip install -q transformers gdown")

add_code([
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "import torchvision.transforms.functional as TF\n",
    "from transformers import VideoMAEForVideoClassification\n",
    "from pathlib import Path\n",
    "from PIL import Image\n",
    "from tqdm.auto import tqdm\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from collections import Counter\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {DEVICE}')"
])

# Config
add_markdown("## 1. Configuration")
add_code([
    "# Model Config\n",
    "MODEL_CKPT = 'MCG-NJU/videomae-base-finetuned-kinetics'\n",
    "NUM_FRAMES = 16\n",
    "IMG_SIZE = 224\n",
    "RESIZE_SIZE = 256\n",
    "MEAN = [0.485, 0.456, 0.406]\n",
    "STD = [0.229, 0.224, 0.225]\n",
    "\n",
    "# Paths\n",
    "PATH_DATA_TRAIN = '/kaggle/input/action-video/data/data_train'\n",
    "PATH_DATA_TEST = '/kaggle/input/action-video/data/test'\n",
    "\n",
    "# Google Drive file ID for best_p1.pt\n",
    "CHECKPOINT_FILE_ID = '190-LrBgEJOyWJQtP0tpvLz0P3TmmxKw7'"
])

# Download checkpoint
add_markdown("## 2. Download Checkpoint")
add_code([
    "# Download best_p1.pt from Google Drive\n",
    "!gdown {CHECKPOINT_FILE_ID} -O best_p1.pt\n",
    "\n",
    "# Download test labels\n",
    "!gdown \"1Xv2CWOqdBj3kt0rkNJKRsodSIEd3-wX_\" -O test_labels.csv -q\n",
    "\n",
    "print('✓ Checkpoint and labels downloaded')"
])

# Dataset classes
add_markdown("## 3. Dataset Classes")
add_code([
    "class VideoDataset(Dataset):\n",
    "    def __init__(self, root, num_frames=16):\n",
    "        self.root = Path(root)\n",
    "        self.num_frames = num_frames\n",
    "        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])\n",
    "        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}\n",
    "        self.samples = []\n",
    "        for cls in self.classes:\n",
    "            cls_dir = self.root / cls\n",
    "            for video_dir in sorted([d for d in cls_dir.iterdir() if d.is_dir()]):\n",
    "                self.samples.append((video_dir, self.class_to_idx[cls]))\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.samples)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        video_dir, label = self.samples[idx]\n",
    "        frame_paths = sorted(video_dir.glob('*.jpg'))\n",
    "        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)\n",
    "        frames = [TF.resize(Image.open(frame_paths[i]).convert('RGB'), RESIZE_SIZE) for i in indices]\n",
    "        processed = [TF.normalize(TF.to_tensor(TF.center_crop(img, IMG_SIZE)), MEAN, STD) for img in frames]\n",
    "        return torch.stack(processed), label\n",
    "\n",
    "class TestDataset(Dataset):\n",
    "    def __init__(self, root, num_frames=16):\n",
    "        self.root = Path(root)\n",
    "        self.num_frames = num_frames\n",
    "        self.samples = sorted([(d, int(d.name)) for d in self.root.iterdir() if d.is_dir()], key=lambda x: x[1])\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.samples)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        video_dir, video_id = self.samples[idx]\n",
    "        frame_paths = sorted(video_dir.glob('*.jpg'))\n",
    "        indices = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)\n",
    "        frames = [TF.resize(Image.open(frame_paths[i]).convert('RGB'), RESIZE_SIZE) for i in indices]\n",
    "        processed = [TF.normalize(TF.to_tensor(TF.center_crop(img, IMG_SIZE)), MEAN, STD) for img in frames]\n",
    "        return torch.stack(processed), video_id\n",
    "\n",
    "print('Dataset classes defined')"
])

# Load data
add_markdown("## 4. Load Data & Model")
add_code([
    "# Load datasets\n",
    "train_dataset = VideoDataset(PATH_DATA_TRAIN, NUM_FRAMES)\n",
    "test_dataset = TestDataset(PATH_DATA_TEST, NUM_FRAMES)\n",
    "\n",
    "# Ground truth\n",
    "gt_df = pd.read_csv('test_labels.csv')\n",
    "gt_dict = dict(zip(gt_df['id'].astype(str), gt_df['class']))\n",
    "\n",
    "# DataLoader\n",
    "test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)\n",
    "\n",
    "print(f'Train samples: {len(train_dataset)}')\n",
    "print(f'Test samples: {len(test_dataset)}')\n",
    "print(f'Classes: {len(train_dataset.classes)}')"
])

add_code([
    "# Load model\n",
    "model = VideoMAEForVideoClassification.from_pretrained(\n",
    "    MODEL_CKPT,\n",
    "    num_labels=len(train_dataset.classes),\n",
    "    ignore_mismatched_sizes=True,\n",
    "    num_frames=NUM_FRAMES\n",
    ").to(DEVICE)\n",
    "\n",
    "# Load checkpoint\n",
    "model.load_state_dict(torch.load('best_p1.pt', map_location=DEVICE))\n",
    "model.eval()\n",
    "\n",
    "print('✓ Model loaded with checkpoint')"
])

# Evaluate
add_markdown("## 5. Evaluate on Test Set")
add_code([
    "# Get predictions\n",
    "all_preds = []\n",
    "all_true = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for videos, video_ids in tqdm(test_loader, desc='Evaluating'):\n",
    "        videos = videos.to(DEVICE)\n",
    "        preds = model(videos).logits.argmax(1).cpu().tolist()\n",
    "        \n",
    "        for vid, pred in zip(video_ids.tolist(), preds):\n",
    "            true_label = gt_dict[str(vid)]\n",
    "            all_true.append(true_label)\n",
    "            all_preds.append(train_dataset.classes[pred])\n",
    "\n",
    "# Overall accuracy\n",
    "from sklearn.metrics import accuracy_score\n",
    "overall_acc = accuracy_score(all_true, all_preds)\n",
    "\n",
    "print(f'\\n' + '=' * 60)\n",
    "print(f'OVERALL TEST ACCURACY: {overall_acc:.4f} ({overall_acc*100:.2f}%)')\n",
    "print('=' * 60)"
])

# Per-class report
add_markdown("## 6. Per-Class Classification Report")
add_code([
    "# Classification report\n",
    "report = classification_report(all_true, all_preds, target_names=train_dataset.classes, \n",
    "                                digits=4, zero_division=0)\n",
    "print(report)\n",
    "\n",
    "# Save report\n",
    "with open('per_class_report.txt', 'w') as f:\n",
    "    f.write(f'Overall Accuracy: {overall_acc:.4f}\\n\\n')\n",
    "    f.write(report)\n",
    "\n",
    "print('\\n✓ Saved to per_class_report.txt')"
])

# Train distribution
add_markdown("## 7. Training Set Class Distribution")
add_code([
    "# Count samples per class in train set\n",
    "class_counts = Counter([label for _, label in train_dataset.samples])\n",
    "class_names = train_dataset.classes\n",
    "\n",
    "# Sort by count\n",
    "sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)\n",
    "sorted_names = [class_names[idx] for idx, _ in sorted_classes]\n",
    "sorted_counts = [count for _, count in sorted_classes]\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(16, 6))\n",
    "bars = plt.bar(range(len(sorted_names)), sorted_counts, color='steelblue', alpha=0.7)\n",
    "plt.xlabel('Action Category', fontsize=12)\n",
    "plt.ylabel('Number of Videos', fontsize=12)\n",
    "plt.title('Training Set Class Distribution', fontsize=14, fontweight='bold')\n",
    "plt.xticks(range(len(sorted_names)), sorted_names, rotation=90, ha='right')\n",
    "plt.grid(axis='y', alpha=0.3)\n",
    "\n",
    "# Highlight extremes\n",
    "bars[0].set_color('darkred')\n",
    "bars[-1].set_color('darkgreen')\n",
    "\n",
    "# Stats\n",
    "max_count = sorted_counts[0]\n",
    "min_count = sorted_counts[-1]\n",
    "ratio = max_count / min_count\n",
    "plt.text(0.02, 0.98, f'Max: {max_count} | Min: {min_count} | Ratio: {ratio:.2f}x', \n",
    "         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',\n",
    "         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('train_distribution.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "print(f'Imbalance ratio: {ratio:.2f}x')"
])

# Per-class accuracy
add_markdown("## 8. Per-Class Test Accuracy")
add_code([
    "# Extract per-class metrics\n",
    "report_dict = classification_report(all_true, all_preds, target_names=train_dataset.classes, \n",
    "                                     output_dict=True, zero_division=0)\n",
    "\n",
    "# Extract recall (accuracy) per class\n",
    "class_accs = [(cls, report_dict[cls]['recall']) for cls in train_dataset.classes]\n",
    "class_accs_sorted = sorted(class_accs, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "cls_names = [c[0] for c in class_accs_sorted]\n",
    "cls_accs_vals = [c[1] * 100 for c in class_accs_sorted]\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(16, 6))\n",
    "colors = ['darkgreen' if acc >= 90 else 'orange' if acc >= 70 else 'darkred' for acc in cls_accs_vals]\n",
    "bars = plt.bar(range(len(cls_names)), cls_accs_vals, color=colors, alpha=0.7)\n",
    "plt.axhline(y=overall_acc*100, color='blue', linestyle='--', linewidth=2, label=f'Overall ({overall_acc*100:.2f}%)')\n",
    "plt.xlabel('Action Category', fontsize=12)\n",
    "plt.ylabel('Accuracy (%)', fontsize=12)\n",
    "plt.title('Per-Class Test Accuracy', fontsize=14, fontweight='bold')\n",
    "plt.xticks(range(len(cls_names)), cls_names, rotation=90, ha='right')\n",
    "plt.ylim(0, 105)\n",
    "plt.legend()\n",
    "plt.grid(axis='y', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
])

# Top/Bottom classes
add_code([
    "print('\\n' + '=' * 60)\n",
    "print('🏆 Top-5 Best Performing Classes:')\n",
    "print('=' * 60)\n",
    "for i in range(min(5, len(cls_names))):\n",
    "    print(f'{i+1:2d}. {cls_names[i]:20s} {cls_accs_vals[i]:6.2f}%')\n",
    "\n",
    "print('\\n' + '=' * 60)\n",
    "print('⚠️  Top-5 Worst Performing Classes:')\n",
    "print('=' * 60)\n",
    "for i in range(max(0, len(cls_names)-5), len(cls_names)):\n",
    "    print(f'{len(cls_names)-i:2d}. {cls_names[i]:20s} {cls_accs_vals[i]:6.2f}%')"
])

# Confusion matrix (top confusions)
add_markdown("## 9. Top Confusion Pairs")
add_code([
    "# Confusion matrix\n",
    "cm = confusion_matrix(all_true, all_preds, labels=train_dataset.classes)\n",
    "\n",
    "# Find top confusions (off-diagonal)\n",
    "confusions = []\n",
    "for i in range(len(train_dataset.classes)):\n",
    "    for j in range(len(train_dataset.classes)):\n",
    "        if i != j and cm[i, j] > 0:\n",
    "            confusions.append((train_dataset.classes[i], train_dataset.classes[j], cm[i, j]))\n",
    "\n",
    "# Sort by confusion count\n",
    "confusions.sort(key=lambda x: x[2], reverse=True)\n",
    "\n",
    "print('\\n' + '=' * 70)\n",
    "print('Top-10 Confusion Pairs (True → Predicted)')\n",
    "print('=' * 70)\n",
    "for i, (true_cls, pred_cls, count) in enumerate(confusions[:10]):\n",
    "    print(f'{i+1:2d}. {true_cls:20s} → {pred_cls:20s} ({int(count):2d} errors)')"
])

# Save summary
add_code([
    "# Save summary stats\n",
    "summary = {\n",
    "    'overall_accuracy': overall_acc,\n",
    "    'best_class': cls_names[0],\n",
    "    'best_acc': cls_accs_vals[0] / 100,\n",
    "    'worst_class': cls_names[-1],\n",
    "    'worst_acc': cls_accs_vals[-1] / 100,\n",
    "    'imbalance_ratio': ratio\n",
    "}\n",
    "\n",
    "pd.DataFrame([summary]).to_csv('analysis_summary.csv', index=False)\n",
    "print('\\n✓ Saved analysis_summary.csv')"
])

# Save notebook
output_path = "/mnt/hdd/Learning/AIO-Projects/Project-7.1/analyze-checkpoint.ipynb"
with open(output_path, 'w') as f:
    json.dump(nb, f, indent=2)

print(f"\n✅ Created: {output_path}")
print("\nNotebook includes:")
print("  - Download checkpoint from Google Drive")
print("  - Per-class accuracy evaluation")
print("  - Training set distribution visualization")
print("  - Per-class test accuracy plot")
print("  - Top confusion pairs analysis")
print("  - Summary statistics export")

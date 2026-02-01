"""
Script to modify project-7-1-phase-3.ipynb for multi-GPU training.
This script adds DataParallel support for training on multiple GPUs.

Run this script once to apply the modifications:
    python scripts/modify_notebook_for_multigpu.py

After running, the notebook will be configured for multi-GPU (2x T4) training.
"""

import json
from pathlib import Path

def modify_notebook():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "project-7-1-phase-3.ipynb"
    
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    modified_count = 0
    
    # ========== MODIFICATION 1: Update imports cell to add multi-GPU detection ==========
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Find the imports cell with device setup
            if 'import torch' in source and "DEVICE = torch.device('cuda'" in source and 'NUM_GPUS' not in source:
                print(f"[Cell {i}] Modifying imports cell to add multi-GPU detection...")
                
                new_source = [
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.nn.functional as F\n",
                    "from torch.utils.data import DataLoader, Dataset\n",
                    "import torchvision.transforms as T\n",
                    "import torchvision.transforms.functional as TF\n",
                    "from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor\n",
                    "from transformers import get_cosine_schedule_with_warmup\n",
                    "from pathlib import Path\n",
                    "from PIL import Image\n",
                    "from tqdm.auto import tqdm\n",
                    "import numpy as np\n",
                    "import random\n",
                    "import os\n",
                    "\n",
                    "# Setup Device & Multi-GPU Configuration\n",
                    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "NUM_GPUS = torch.cuda.device_count()\n",
                    "USE_MULTI_GPU = NUM_GPUS > 1\n",
                    "print(f'Using device: {DEVICE}')\n",
                    "print(f'Number of GPUs available: {NUM_GPUS}')\n",
                    "if USE_MULTI_GPU:\n",
                    "    print(f'Multi-GPU training ENABLED with DataParallel')\n",
                    "\n",
                    "# Paths (Keep original paths)\n",
                    "PATH_DATA_TRAIN = r'/kaggle/input/action-video/data/data_train'\n",
                    "PATH_DATA_TEST = r'/kaggle/input/action-video/data/test'\n"
                ]
                
                cell['source'] = new_source
                cell['outputs'] = []  # Clear old outputs
                modified_count += 1
                break
    
    # ========== MODIFICATION 2: Update model loading cell to wrap with DataParallel ==========
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Find the model loading cell
            if 'print("Loading VideoMAE...")' in source and 'VideoMAEForVideoClassification.from_pretrained' in source and 'DataParallel' not in source:
                print(f"[Cell {i}] Modifying model loading cell to add DataParallel wrapper...")
                
                new_source = [
                    'print("Loading VideoMAE...")\n',
                    'model = VideoMAEForVideoClassification.from_pretrained(\n',
                    '    MODEL_CKPT,\n',
                    '    label2id=label2id,\n',
                    '    id2label=id2label,\n',
                    '    ignore_mismatched_sizes=True, \n',
                    '    num_frames=NUM_FRAMES\n',
                    ')\n',
                    '\n',
                    '# Move to device and wrap with DataParallel if multiple GPUs\n',
                    'model.to(DEVICE)\n',
                    'if USE_MULTI_GPU:\n',
                    '    model = nn.DataParallel(model)\n',
                    '    print(f"Model wrapped with DataParallel across {NUM_GPUS} GPUs")\n',
                    'print("Model loaded.")'
                ]
                
                cell['source'] = new_source
                cell['outputs'] = []  # Clear old outputs
                modified_count += 1
                break
    
    # ========== MODIFICATION 3: Update training execution cell to handle DataParallel save/load ==========
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Find the training execution cell (Phase 1 & Phase 2)
            if '# ---------------- PHASE 1: TRAINING WITH MIXUP ----------------' in source and 'get_underlying_model' not in source:
                print(f"[Cell {i}] Modifying training execution cell for DataParallel-aware save/load...")
                
                new_source = [
                    '# Helper function to get underlying model (handles DataParallel wrapper)\n',
                    'def get_underlying_model(model):\n',
                    '    """Returns the underlying model, unwrapping DataParallel if necessary."""\n',
                    '    return model.module if isinstance(model, nn.DataParallel) else model\n',
                    '\n',
                    '# ---------------- PHASE 1: TRAINING WITH MIXUP ----------------\n',
                    'print("\\n" + "="*40)\n',
                    'print(f"STARTING PHASE 1 (Mixup Enabled, LR={LR_P1}, Epochs={EPOCHS_P1})")\n',
                    'if USE_MULTI_GPU:\n',
                    '    print(f"Training on {NUM_GPUS} GPUs with DataParallel")\n',
                    'print("="*40)\n',
                    '\n',
                    'optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P1, weight_decay=WEIGHT_DECAY)\n',
                    'scaler = torch.amp.GradScaler(enabled=True)\n',
                    'num_training_steps = len(train_loader_p1) * EPOCHS_P1 // ACCUM_STEPS\n',
                    'num_warmup_steps = int(num_training_steps * WARMUP_RATIO)\n',
                    'scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)\n',
                    '\n',
                    'best_acc_p1 = 0.0\n',
                    '\n',
                    'for epoch in range(EPOCHS_P1):\n',
                    '    print(f"\\nEpoch {epoch+1}/{EPOCHS_P1} (Phase 1)")\n',
                    '    loss, acc = train_epoch(model, train_loader_p1, optimizer, scheduler, scaler, DEVICE, ACCUM_STEPS, use_mixup=True)\n',
                    '    print(f"  Result: Loss = {loss:.4f} | Acc = {acc:.4f}")\n',
                    '    \n',
                    '    if acc > best_acc_p1:\n',
                    '        best_acc_p1 = acc\n',
                    '        # Save underlying model (unwrap DataParallel if used)\n',
                    '        get_underlying_model(model).save_pretrained("./videomae_phase1_best")\n',
                    '        print(f"  >>> Saved Phase 1 Best (Acc: {best_acc_p1:.4f})")\n',
                    '\n',
                    '# ---------------- PHASE 2: FINE-TUNING (NO MIXUP + LABEL SMOOTHING) ----------------\n',
                    'print("\\n" + "="*40)\n',
                    'print(f"STARTING PHASE 2 (No Mixup, Label Smooth={LABEL_SMOOTHING}, Low LR={LR_P2})")\n',
                    'if USE_MULTI_GPU:\n',
                    '    print(f"Training on {NUM_GPUS} GPUs with DataParallel")\n',
                    'print("="*40)\n',
                    '\n',
                    '# Load best model from Phase 1\n',
                    'print("Loading best model from Phase 1...")\n',
                    'model = VideoMAEForVideoClassification.from_pretrained(\n',
                    '    "./videomae_phase1_best",\n',
                    '    label2id=label2id,\n',
                    '    id2label=id2label,\n',
                    '    num_frames=NUM_FRAMES,\n',
                    '    ignore_mismatched_sizes=False \n',
                    ').to(DEVICE)\n',
                    '\n',
                    '# Wrap with DataParallel if multiple GPUs available\n',
                    'if USE_MULTI_GPU:\n',
                    '    model = nn.DataParallel(model)\n',
                    '    print(f"Model wrapped with DataParallel across {NUM_GPUS} GPUs")\n',
                    '\n',
                    '# Re-init Optimizer with LOW LR\n',
                    'optimizer = torch.optim.AdamW(model.parameters(), lr=LR_P2, weight_decay=WEIGHT_DECAY)\n',
                    'scaler = torch.amp.GradScaler(enabled=True)\n',
                    'num_training_steps = len(train_loader_p2) * EPOCHS_P2 // ACCUM_STEPS\n',
                    'scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps) \n',
                    '\n',
                    'best_acc_p2 = best_acc_p1\n',
                    '\n',
                    'for epoch in range(EPOCHS_P2):\n',
                    '    print(f"\\nEpoch {epoch+1}/{EPOCHS_P2} (Phase 2)")\n',
                    '    # Pass label_smoothing param here\n',
                    '    loss, acc = train_epoch(model, train_loader_p2, optimizer, scheduler, scaler, DEVICE, ACCUM_STEPS, use_mixup=False, label_smoothing=LABEL_SMOOTHING)\n',
                    '    print(f"  Result: Loss = {loss:.4f} | Acc = {acc:.4f}")\n',
                    '    \n',
                    '    if acc > best_acc_p2:\n',
                    '        best_acc_p2 = acc\n',
                    '        # Save underlying model (unwrap DataParallel if used)\n',
                    '        get_underlying_model(model).save_pretrained("./videomae_final_best")\n',
                    '        print(f"  >>> Saved Phase 2 Best (Acc: {best_acc_p2:.4f})")\n',
                    '    else:\n',
                    '        if epoch == EPOCHS_P2 - 1:\n',
                    '             get_underlying_model(model).save_pretrained("./videomae_final_last")\n',
                    '             print("  >>> Saved Last Model")'
                ]
                
                cell['source'] = new_source
                cell['outputs'] = []  # Clear old outputs
                modified_count += 1
                break
    
    # Save the modified notebook
    print(f"\nTotal cells modified: {modified_count}")
    
    if modified_count > 0:
        # Create backup
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(notebook_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())
        print(f"Backup saved to: {backup_path}")
        
        # Save modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Modified notebook saved to: {notebook_path}")
        print("\n✅ Successfully updated notebook for multi-GPU training!")
        print("The notebook will now use DataParallel when 2+ GPUs are detected.")
    else:
        print("⚠️ No cells were modified. The notebook may already be configured for multi-GPU.")

if __name__ == "__main__":
    modify_notebook()

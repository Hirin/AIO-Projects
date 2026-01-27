"""
Add offline augmentation cells to videoMAE-complete.ipynb
"""
import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/project-7.1.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find index after "## 1. Configuration" cell
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if '## 2. Dataset Classes' in source:
            insert_idx = i
            break

if insert_idx is None:
    print("Error: Could not find insertion point")
    exit(1)

print(f"Found insertion point at cell {insert_idx}")

# New cells to insert
new_cells = []

# Cell 1: Markdown header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1.5. Offline Data Augmentation\n",
        "\n",
        "**Strategy**: Balance class distribution by generating 6 variations:\n",
        "1. Original\n",
        "2. Horizontal Flip\n",
        "3. Rotate ±10°\n",
        "4. Crop/Zoom (90%)\n",
        "5. Flip + Brightness\n",
        "6. Rotate + Crop"
    ]
})

# Cell 2: Augmentation functions
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "import shutil\n",
        "from collections import Counter\n",
        "\n",
        "def get_class_distribution(train_path):\n",
        "    \"\"\"Count samples per class\"\"\"\n",
        "    train_path = Path(train_path)\n",
        "    class_counts = {}\n",
        "    for cls_dir in train_path.iterdir():\n",
        "        if cls_dir.is_dir():\n",
        "            video_count = len([d for d in cls_dir.iterdir() if d.is_dir()])\n",
        "            class_counts[cls_dir.name] = video_count\n",
        "    return class_counts\n",
        "\n",
        "def augment_video_frames(src_dir, dst_dir, transform_type):\n",
        "    \"\"\"Apply augmentation to all frames in a video directory\"\"\"\n",
        "    dst_dir.mkdir(parents=True, exist_ok=True)\n",
        "    \n",
        "    for frame_path in sorted(src_dir.glob('*.jpg')):\n",
        "        img = Image.open(frame_path)\n",
        "        \n",
        "        if transform_type == 'flip':\n",
        "            img = TF.hflip(img)\n",
        "        elif transform_type == 'rotate_neg':\n",
        "            img = TF.rotate(img, -10, fill=0)\n",
        "        elif transform_type == 'rotate_pos':\n",
        "            img = TF.rotate(img, 10, fill=0)\n",
        "        elif transform_type == 'crop':\n",
        "            w, h = img.size\n",
        "            crop_size = int(min(w, h) * 0.9)\n",
        "            left = (w - crop_size) // 2\n",
        "            top = (h - crop_size) // 2\n",
        "            img = TF.crop(img, top, left, crop_size, crop_size)\n",
        "            img = TF.resize(img, (h, w))\n",
        "        elif transform_type == 'flip_bright':\n",
        "            img = TF.hflip(img)\n",
        "            img = TF.adjust_brightness(img, 1.2)\n",
        "        elif transform_type == 'rotate_crop':\n",
        "            img = TF.rotate(img, -8, fill=0)\n",
        "            w, h = img.size\n",
        "            crop_size = int(min(w, h) * 0.92)\n",
        "            left = (w - crop_size) // 2\n",
        "            top = (h - crop_size) // 2\n",
        "            img = TF.crop(img, top, left, crop_size, crop_size)\n",
        "            img = TF.resize(img, (h, w))\n",
        "        \n",
        "        img.save(dst_dir / frame_path.name, quality=95)\n",
        "\n",
        "def balance_dataset(train_path, output_path, target_per_class=None):\n",
        "    \"\"\"Balance dataset by augmenting minority classes\"\"\"\n",
        "    train_path = Path(train_path)\n",
        "    output_path = Path(output_path)\n",
        "    \n",
        "    # Get current distribution\n",
        "    class_counts = get_class_distribution(train_path)\n",
        "    max_count = max(class_counts.values())\n",
        "    target = target_per_class or max_count\n",
        "    \n",
        "    print(f'Target samples per class: {target}')\n",
        "    print(f'Max class: {max_count}, Min class: {min(class_counts.values())}')\n",
        "    \n",
        "    # Augmentation types\n",
        "    aug_types = ['flip', 'rotate_neg', 'rotate_pos', 'crop', 'flip_bright', 'rotate_crop']\n",
        "    \n",
        "    # Copy original + augment\n",
        "    total_created = 0\n",
        "    for cls_name, count in tqdm(class_counts.items(), desc='Balancing classes'):\n",
        "        cls_src = train_path / cls_name\n",
        "        cls_dst = output_path / cls_name\n",
        "        cls_dst.mkdir(parents=True, exist_ok=True)\n",
        "        \n",
        "        videos = sorted([d for d in cls_src.iterdir() if d.is_dir()])\n",
        "        \n",
        "        # Copy originals\n",
        "        for v in videos:\n",
        "            dst = cls_dst / v.name\n",
        "            if not dst.exists():\n",
        "                shutil.copytree(v, dst)\n",
        "        \n",
        "        # Calculate how many augmented samples needed\n",
        "        needed = target - count\n",
        "        if needed <= 0:\n",
        "            continue\n",
        "        \n",
        "        # Generate augmented samples\n",
        "        aug_idx = 0\n",
        "        created = 0\n",
        "        while created < needed:\n",
        "            for v in videos:\n",
        "                if created >= needed:\n",
        "                    break\n",
        "                aug_type = aug_types[aug_idx % len(aug_types)]\n",
        "                aug_name = f'{v.name}_aug_{aug_type}_{aug_idx // len(aug_types)}'\n",
        "                dst = cls_dst / aug_name\n",
        "                if not dst.exists():\n",
        "                    augment_video_frames(v, dst, aug_type)\n",
        "                    created += 1\n",
        "                    total_created += 1\n",
        "            aug_idx += 1\n",
        "    \n",
        "    print(f'\\n✓ Created {total_created} augmented videos')\n",
        "    return output_path\n",
        "\n",
        "print('Augmentation functions defined')"
    ]
})

# Cell 3: Show before distribution
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Analyze BEFORE augmentation\n",
        "print('=' * 60)\n",
        "print('BEFORE AUGMENTATION')\n",
        "print('=' * 60)\n",
        "\n",
        "before_counts = get_class_distribution(PATH_DATA_TRAIN)\n",
        "before_df = pd.DataFrame([\n",
        "    {'class': k, 'count': v} for k, v in before_counts.items()\n",
        "]).sort_values('count', ascending=False)\n",
        "\n",
        "print(f'Total classes: {len(before_counts)}')\n",
        "print(f'Total samples: {sum(before_counts.values())}')\n",
        "print(f'Max: {max(before_counts.values())} ({max(before_counts, key=before_counts.get)})')\n",
        "print(f'Min: {min(before_counts.values())} ({min(before_counts, key=before_counts.get)})')\n",
        "print(f'Imbalance ratio: {max(before_counts.values()) / min(before_counts.values()):.2f}x')\n",
        "\n",
        "# Plot\n",
        "plt.figure(figsize=(16, 5))\n",
        "plt.bar(range(len(before_df)), before_df['count'].values, color='steelblue', alpha=0.7)\n",
        "plt.xlabel('Class (sorted by count)')\n",
        "plt.ylabel('Number of samples')\n",
        "plt.title('BEFORE Augmentation: Class Distribution')\n",
        "plt.xticks(range(len(before_df)), before_df['class'].values, rotation=90, fontsize=7)\n",
        "plt.tight_layout()\n",
        "plt.savefig('distribution_before.png', dpi=150)\n",
        "plt.show()"
    ]
})

# Cell 4: Run augmentation
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Run augmentation\n",
        "PATH_DATA_AUGMENTED = '/kaggle/working/data_train_augmented'\n",
        "\n",
        "# Balance to max class count\n",
        "balanced_path = balance_dataset(PATH_DATA_TRAIN, PATH_DATA_AUGMENTED)\n",
        "\n",
        "# Update train path for rest of notebook\n",
        "PATH_DATA_TRAIN = str(balanced_path)\n",
        "print(f'\\n✓ Updated PATH_DATA_TRAIN to: {PATH_DATA_TRAIN}')"
    ]
})

# Cell 5: Show after distribution
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Analyze AFTER augmentation\n",
        "print('=' * 60)\n",
        "print('AFTER AUGMENTATION')\n",
        "print('=' * 60)\n",
        "\n",
        "after_counts = get_class_distribution(PATH_DATA_TRAIN)\n",
        "after_df = pd.DataFrame([\n",
        "    {'class': k, 'count': v} for k, v in after_counts.items()\n",
        "]).sort_values('count', ascending=False)\n",
        "\n",
        "print(f'Total classes: {len(after_counts)}')\n",
        "print(f'Total samples: {sum(after_counts.values())}')\n",
        "print(f'Max: {max(after_counts.values())} ({max(after_counts, key=after_counts.get)})')\n",
        "print(f'Min: {min(after_counts.values())} ({min(after_counts, key=after_counts.get)})')\n",
        "print(f'Imbalance ratio: {max(after_counts.values()) / min(after_counts.values()):.2f}x')\n",
        "\n",
        "# Plot\n",
        "plt.figure(figsize=(16, 5))\n",
        "plt.bar(range(len(after_df)), after_df['count'].values, color='darkgreen', alpha=0.7)\n",
        "plt.xlabel('Class (sorted by count)')\n",
        "plt.ylabel('Number of samples')\n",
        "plt.title('AFTER Augmentation: Class Distribution (Balanced)')\n",
        "plt.xticks(range(len(after_df)), after_df['class'].values, rotation=90, fontsize=7)\n",
        "plt.tight_layout()\n",
        "plt.savefig('distribution_after.png', dpi=150)\n",
        "plt.show()\n",
        "\n",
        "# Summary comparison\n",
        "print('\\n' + '=' * 60)\n",
        "print('SUMMARY COMPARISON')\n",
        "print('=' * 60)\n",
        "print(f'Before: {sum(before_counts.values())} samples, ratio: {max(before_counts.values())/min(before_counts.values()):.2f}x')\n",
        "print(f'After:  {sum(after_counts.values())} samples, ratio: {max(after_counts.values())/min(after_counts.values()):.2f}x')\n",
        "print(f'Added:  {sum(after_counts.values()) - sum(before_counts.values())} augmented samples')"
    ]
})

# Cell 6: Demo augmented samples
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Demo: Show augmented samples\n",
        "def show_augmentation_demo(train_path, class_name='smile', num_samples=3):\n",
        "    \"\"\"Show original vs augmented samples\"\"\"\n",
        "    train_path = Path(train_path)\n",
        "    cls_dir = train_path / class_name\n",
        "    \n",
        "    if not cls_dir.exists():\n",
        "        print(f'Class {class_name} not found')\n",
        "        return\n",
        "    \n",
        "    # Find one original and its augmented versions\n",
        "    videos = sorted([d for d in cls_dir.iterdir() if d.is_dir()])\n",
        "    originals = [v for v in videos if '_aug_' not in v.name][:num_samples]\n",
        "    \n",
        "    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))\n",
        "    \n",
        "    aug_labels = ['Original', 'Flip', 'Rotate-', 'Crop', 'Flip+Bright', 'Rot+Crop']\n",
        "    \n",
        "    for row, orig in enumerate(originals):\n",
        "        # Get first frame of each version\n",
        "        for col, label in enumerate(aug_labels):\n",
        "            if col == 0:\n",
        "                video_dir = orig\n",
        "            else:\n",
        "                aug_suffix = ['flip', 'rotate_neg', 'crop', 'flip_bright', 'rotate_crop'][col-1]\n",
        "                aug_name = f'{orig.name}_aug_{aug_suffix}_0'\n",
        "                video_dir = cls_dir / aug_name\n",
        "            \n",
        "            if video_dir.exists():\n",
        "                frame = sorted(video_dir.glob('*.jpg'))[0]\n",
        "                img = Image.open(frame)\n",
        "                axes[row, col].imshow(img)\n",
        "            else:\n",
        "                axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center')\n",
        "            \n",
        "            axes[row, col].axis('off')\n",
        "            if row == 0:\n",
        "                axes[row, col].set_title(label, fontsize=10)\n",
        "    \n",
        "    plt.suptitle(f'Augmentation Demo: {class_name}', fontsize=14, fontweight='bold')\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('augmentation_demo.png', dpi=150, bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "# Show demo for a minority class\n",
        "minority_class = min(before_counts, key=before_counts.get)\n",
        "print(f'Showing augmentation demo for minority class: {minority_class}')\n",
        "show_augmentation_demo(PATH_DATA_TRAIN, minority_class, num_samples=2)"
    ]
})

# Insert new cells before "## 2. Dataset Classes"
for i, cell in enumerate(new_cells):
    nb['cells'].insert(insert_idx + i, cell)

print(f"Inserted {len(new_cells)} new cells")

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print(f"\n✅ Successfully updated {notebook_path}")
print("\nNew cells added:")
print("  1. Markdown: Offline Data Augmentation header")
print("  2. Code: Augmentation functions (flip, rotate, crop, brightness)")
print("  3. Code: Analyze BEFORE distribution + plot")
print("  4. Code: Run augmentation to balance classes")
print("  5. Code: Analyze AFTER distribution + plot + comparison")
print("  6. Code: Demo augmented samples visualization")

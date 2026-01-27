"""
Add per-class analysis cells at the end of videoMAE-complete.ipynb
"""
import json

notebook_path = '/mnt/hdd/Learning/AIO-Projects/Project-7.1/project-7.1.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cells to append at the end
new_cells = []

# Cell 1: Markdown header
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Per-Class Accuracy Analysis\n",
        "\n",
        "Analyze model performance on each class after training."
    ]
})

# Cell 2: Per-class evaluation
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Load best model and evaluate per-class\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "import seaborn as sns\n",
        "\n",
        "# Load best checkpoint\n",
        "if os.path.exists('best_final.pt'):\n",
        "    model.load_state_dict(torch.load('best_final.pt'))\n",
        "    print('Loaded best_final.pt')\n",
        "elif os.path.exists('best_p1.pt'):\n",
        "    model.load_state_dict(torch.load('best_p1.pt'))\n",
        "    print('Loaded best_p1.pt')\n",
        "model.eval()\n",
        "\n",
        "# Get predictions\n",
        "all_preds = []\n",
        "all_true = []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for videos, video_ids in tqdm(test_loader, desc='Evaluating per-class'):\n",
        "        videos = videos.to(DEVICE)\n",
        "        preds = model(videos).logits.argmax(1).cpu().tolist()\n",
        "        \n",
        "        for vid, pred in zip(video_ids.tolist(), preds):\n",
        "            true_label = gt_dict[str(vid)]\n",
        "            all_true.append(true_label)\n",
        "            all_preds.append(train_dataset.classes[pred])\n",
        "\n",
        "# Overall accuracy\n",
        "overall_acc = accuracy_score(all_true, all_preds)\n",
        "print(f'\\n{\"=\"*60}')\n",
        "print(f'OVERALL TEST ACCURACY: {overall_acc:.4f} ({overall_acc*100:.2f}%)')\n",
        "print(f'{\"=\"*60}')"
    ]
})

# Cell 3: Classification report
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Full classification report\n",
        "print('\\nPER-CLASS CLASSIFICATION REPORT:')\n",
        "print('=' * 80)\n",
        "report = classification_report(all_true, all_preds, target_names=train_dataset.classes, \n",
        "                                digits=4, zero_division=0)\n",
        "print(report)\n",
        "\n",
        "# Save report\n",
        "with open('per_class_report.txt', 'w') as f:\n",
        "    f.write(f'Overall Accuracy: {overall_acc:.4f}\\n\\n')\n",
        "    f.write(report)\n",
        "print('\\n✓ Saved to per_class_report.txt')"
    ]
})

# Cell 4: Per-class accuracy plot
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Extract per-class accuracy and visualize\n",
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
        "plt.title('Per-Class Test Accuracy (Sorted by Performance)', fontsize=14, fontweight='bold')\n",
        "plt.xticks(range(len(cls_names)), cls_names, rotation=90, ha='right', fontsize=8)\n",
        "plt.ylim(0, 105)\n",
        "plt.legend()\n",
        "plt.grid(axis='y', alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]
})

# Cell 5: Top/Bottom classes
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Top and Bottom performing classes\n",
        "print('\\n' + '=' * 60)\n",
        "print('🏆 TOP-5 BEST PERFORMING CLASSES:')\n",
        "print('=' * 60)\n",
        "for i in range(min(5, len(cls_names))):\n",
        "    print(f'{i+1:2d}. {cls_names[i]:20s} {cls_accs_vals[i]:6.2f}%')\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('⚠️  TOP-5 WORST PERFORMING CLASSES:')\n",
        "print('=' * 60)\n",
        "for i in range(max(0, len(cls_names)-5), len(cls_names)):\n",
        "    print(f'{len(cls_names)-i:2d}. {cls_names[i]:20s} {cls_accs_vals[i]:6.2f}%')"
    ]
})

# Cell 6: Confusion analysis
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Top confusion pairs\n",
        "cm = confusion_matrix(all_true, all_preds, labels=train_dataset.classes)\n",
        "\n",
        "# Find top confusions (off-diagonal)\n",
        "confusions = []\n",
        "for i in range(len(train_dataset.classes)):\n",
        "    for j in range(len(train_dataset.classes)):\n",
        "        if i != j and cm[i, j] > 0:\n",
        "            confusions.append((train_dataset.classes[i], train_dataset.classes[j], cm[i, j]))\n",
        "\n",
        "confusions.sort(key=lambda x: x[2], reverse=True)\n",
        "\n",
        "print('\\n' + '=' * 70)\n",
        "print('TOP-10 CONFUSION PAIRS (True → Predicted)')\n",
        "print('=' * 70)\n",
        "for i, (true_cls, pred_cls, count) in enumerate(confusions[:10]):\n",
        "    print(f'{i+1:2d}. {true_cls:20s} → {pred_cls:20s} ({int(count):2d} errors)')"
    ]
})

# Cell 7: Final summary
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# Final summary\n",
        "print('\\n' + '=' * 70)\n",
        "print('FINAL TRAINING SUMMARY')\n",
        "print('=' * 70)\n",
        "print(f'Overall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)')\n",
        "print(f'Best performing class: {cls_names[0]} ({cls_accs_vals[0]:.2f}%)')\n",
        "print(f'Worst performing class: {cls_names[-1]} ({cls_accs_vals[-1]:.2f}%)')\n",
        "print(f'Classes with 100% accuracy: {sum(1 for acc in cls_accs_vals if acc >= 99.9)}')\n",
        "print(f'Classes with <70% accuracy: {sum(1 for acc in cls_accs_vals if acc < 70)}')\n",
        "\n",
        "# Save summary\n",
        "summary = {\n",
        "    'overall_accuracy': overall_acc,\n",
        "    'best_class': cls_names[0],\n",
        "    'best_acc': cls_accs_vals[0] / 100,\n",
        "    'worst_class': cls_names[-1],\n",
        "    'worst_acc': cls_accs_vals[-1] / 100,\n",
        "}\n",
        "pd.DataFrame([summary]).to_csv('analysis_summary.csv', index=False)\n",
        "print('\\n✓ Saved analysis_summary.csv')"
    ]
})

# Append cells to notebook
nb['cells'].extend(new_cells)

print(f"Appended {len(new_cells)} new cells")

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print(f"\n✅ Successfully updated {notebook_path}")
print("\nNew cells added at the end:")
print("  1. Per-class evaluation setup")
print("  2. Classification report")
print("  3. Per-class accuracy bar plot")
print("  4. Top/Bottom 5 classes")
print("  5. Confusion pairs analysis")
print("  6. Final summary export")

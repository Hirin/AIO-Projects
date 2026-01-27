#!/usr/bin/env python3
"""Update dual-GPU notebook to add per-epoch test evaluation and plots."""

import json

def main():
    with open('ablation-study-dual-gpu.ipynb', 'r') as f:
        nb = json.load(f)
    
    # Find and update the run_single_experiment function (cell 10)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def run_single_experiment' in source:
                cell['source'] = [
'''## 10. Single GPU Experiment Runner (with per-epoch test eval)

def run_single_experiment(exp_config, gpu_id, results_queue):
    """Run a single experiment on specified GPU with per-epoch test eval."""
    import matplotlib.pyplot as plt
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    exp_name = exp_config['name']
    print(f"[GPU {gpu_id}] Starting: {exp_name}")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create model
    if exp_config.get('is_vit', False):
        model = LightweightViTForAction(num_classes=51).to(device)
        is_vit = True
    else:
        label2id = exp_config['train_ds'].class_to_idx
        id2label = {v: k for k, v in label2id.items()}
        model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_CKPT, label2id=label2id, id2label=id2label,
            ignore_mismatched_sizes=True, num_frames=NUM_FRAMES
        ).to(device)
        is_vit = False
    
    train_ds = exp_config['train_ds']
    test_ds = exp_config['test_ds']
    label2id = train_ds.class_to_idx
    id2label = {v: k for k, v in label2id.items()}
    
    # DataLoaders
    collate_fn = exp_config.get('mixup_collate', None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4 if exp_config.get('multi_view') else BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()
    num_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(num_steps * WARMUP_RATIO), num_steps)
    
    # Training history
    history = {'epoch': [], 'loss_train': [], 'acc_train': [], 'acc_test': []}
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        loss, acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            use_mixup=exp_config.get('use_mixup', False),
            label_smoothing=exp_config.get('label_smoothing', 0.0),
            is_vit=is_vit
        )
        
        # Evaluate on test
        predictions = evaluate(model, test_loader, device, 
                               multi_view=exp_config.get('multi_view', False), 
                               id2label=id2label, is_vit=is_vit)
        test_acc = calc_accuracy(predictions)
        
        print(f"[GPU {gpu_id}] {exp_name} Epoch {epoch+1}/{EPOCHS}: Loss_train={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
        
        history['epoch'].append(epoch + 1)
        history['loss_train'].append(loss)
        history['acc_train'].append(acc)
        history['acc_test'].append(test_acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{exp_name}_best.pt')
    
    # 2-Stage Phase 2
    if exp_config.get('two_stage', False):
        print(f"[GPU {gpu_id}] {exp_name} Phase 2...")
        model.load_state_dict(torch.load(f'{exp_name}_best.pt'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=WEIGHT_DECAY)
        p2_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(p2_loader) * 3 // GRAD_ACCUM_STEPS)
        for epoch in range(3):
            loss, acc = train_epoch(model, p2_loader, optimizer, scheduler, scaler, device, label_smoothing=0.1, is_vit=is_vit)
            predictions = evaluate(model, test_loader, device, multi_view=exp_config.get('multi_view', False), id2label=id2label, is_vit=is_vit)
            test_acc = calc_accuracy(predictions)
            print(f"[GPU {gpu_id}] {exp_name} P2 Epoch {epoch+1}/3: Loss_train={loss:.4f}, Acc_train={acc:.4f}, Acc_test={test_acc:.4f}")
            history['epoch'].append(EPOCHS + epoch + 1)
            history['loss_train'].append(loss)
            history['acc_train'].append(acc)
            history['acc_test'].append(test_acc)
            best_acc = max(best_acc, acc)
    else:
        model.load_state_dict(torch.load(f'{exp_name}_best.pt'))
    
    # Final test
    final_test_acc = history['acc_test'][-1]
    
    # Plot training curves for this experiment
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax1.plot(history['epoch'], history['loss_train'], 'b-o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{exp_name} - Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(history['epoch'], [a*100 for a in history['acc_train']], 'b-o', label='Train', markersize=4)
    ax2.plot(history['epoch'], [a*100 for a in history['acc_test']], 'r-s', label='Test', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{exp_name} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{exp_name} Training Curves (GPU {gpu_id})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"[GPU {gpu_id}] {exp_name} >>> FINAL TEST ACC: {final_test_acc:.4f}")
    
    results_queue.put({
        'exp': exp_name, 
        'train_acc': best_acc, 
        'test_acc': final_test_acc, 
        'gpu': gpu_id,
        'history': history
    })
    
    del model
    torch.cuda.empty_cache()
'''
                ]
                print(f'Updated cell {i} with per-epoch test eval')
                break
    
    with open('ablation-study-dual-gpu.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print('Saved notebook')

if __name__ == '__main__':
    main()

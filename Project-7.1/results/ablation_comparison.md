# VideoMAE Ablation Study Results

## Summary

Results extracted from notebooks after 12h Kaggle session.

---

## 8-Frame Results (COMPLETE)

| Experiment | Test Accuracy | Notes |
|------------|--------------|-------|
| **8frames_baseline** | **81.96%** | Paper baseline, 10 epochs |
| **8frames_custom_full** | **82.55%** | Consistent + Mixup + 2-Stage + Flip TTA |
| **Improvement** | **+0.59%** | Custom vs Baseline |

**Training Details:**
- Batch size: 40
- NUM_FRAMES: 8 (less duplication with 7-15 frame videos)
- Total time: ~2h (parallel on 2 GPUs)

**Key Findings:**
- 8-frame baseline performs **much better** than 16-frame (81.96% vs 69.22%)
- Less frame duplication → better learning
- Custom improvements add +0.59% on top

---

## 16-Frame Results (INCOMPLETE - Session Expired)

| Experiment | Test Accuracy | Status |
|------------|--------------|--------|
| **Exp1_VideoMAE_Paper** | **69.22%** | ✅ Complete |
| Other experiments | ? | ❌ Lost (session expired) |

**Notes:**
- Session expired before completing all 8 experiments
- Only got baseline result
- Significantly worse than 8-frame (69.22% vs 81.96%)

---

## Key Insights

### 1. **8 frames >> 16 frames for this dataset**
```
8f baseline:  81.96%
16f baseline: 69.22%
Difference:   +12.74%
```

**Why?**
- Videos only have 7-15 frames
- 16-frame sampling causes heavy duplication
- 8-frame captures more unique information

### 2. **Custom improvements work**
```
8f baseline: 81.96%
8f custom:   82.55%
Gain:        +0.59%
```

**Techniques:**
- Consistent spatial transforms
- Mixup regularization
- 2-Stage training
- Flip TTA

---

## Recommendations

1. **Use NUM_FRAMES=8** for this dataset (short videos)
2. **Continue ablation with 8-frame** to test individual improvements
3. **Target:** Reproduce 87% from `project-7-1-phase-3.ipynb`

---

## Files

- Source: `8-frames.ipynb` (complete)
- Source: `ablation-study_v2.ipynb` (incomplete)
- Extracted: `results/ablation_results_extracted.csv`

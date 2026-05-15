# Deployment Failures & Lessons

## Overview

AIMO3 involved numerous deployment challenges. This document chronicles the major failures and insights.

## Failure 1: LoRA Fine-Tuning Quantization Mismatch

### The Problem
Trained a LoRA adapter for GPT-OSS-120B using 8-bit quantization (QLoRA):
- Training loss: converged smoothly
- Training accuracy: 92%
- But: deployment crashed with CUDA errors

### Root Cause
- Model was loaded with `load_in_4bit=True` (NXFP4 quantization)
- LoRA adapter was trained on 8-bit (INT8) precision
- Adapter weight shape mismatch: (r, hidden) vs (r, hidden_quantized)
- vLLM automatically requantized, losing adapter fidelity

### The Fix
- Retrained with `load_in_4bit=True` instead of 8-bit
- Verified adapter weight dtype and shape at load time
- But: training time increased 3x due to slower 4-bit operations
- Decision: Abandoned fine-tuning, used base model instead

### Lesson
**Training precision must match deployment precision.**
- Mismatches silently fail at inference time
- Adapter weights become corrupted during quantization
- Always validate weight shapes and dtypes across precision changes

---

## Failure 2: BF16 Memory Explosion

### The Problem
Attempted to reduce quantization overhead by loading in BF16 (bfloat16) instead of INT4:
- Model loaded successfully: 60 GB VRAM used
- First batch of 4 problems: processed correctly
- Batch 5: OOM (Out of Memory) error

### Root Cause
KV cache accumulation in BF16:
- Each token in attention cache: 2x larger than INT4
- 8 parallel sequences × 2000 tokens × 2x precision = ~64 GB KV cache
- Total: 60 GB weights + 64 GB cache = 124 GB (exceeds H100 80GB)

vLLM's paging system ineffective because:
- No CPU offloading configured
- Batching algorithm didn't account for KV expansion ratio
- Block size too large for efficient reuse

### The Fix
Multiple solutions attempted:

**Option A: Reduce sequence length**
- Limited output to max 512 tokens
- Freed ~20 GB, but truncated reasoning
- Trade-off: Score dropped 2-3 points

**Option B: Enable CPU offloading**
- PCIe bandwidth became bottleneck
- Latency increased from 4s to 12s per problem
- Infeasible for competition deadline

**Option C: Quantize to INT4**
- Reduced KV cache back to manageable size
- Solved OOM
- Accepted ~0.5-1.0 point accuracy loss from quantization

### Lesson
**Memory profiling must include KV cache, not just weights.**
- KV cache can exceed weight memory in long-sequence scenarios
- Precision changes have cascading effects on cache size
- Quantization trade-off: 1-2% accuracy loss vs OOM crash

---

## Failure 3: Entropy Weighting NaN Propagation

### The Problem
Entropy-weighted voting occasionally produced NaN (Not-a-Number) in final predictions:
- 1-2 problems per 100 produced NaN answers
- Competition submitter caught this and defaulted to 0
- Cost: -5 to -10 points per NaN case

### Root Cause
Logarithm of zero in entropy computation:
```python
entropy = -sum(p * log(p))  # When p=0, log(0) = -inf
```

When a model produced empty output or very short output:
- Token probability distribution sparse
- Some probabilities exactly 0.0 (due to quantization)
- log(0) = -inf, entropy = NaN
- Weight = exp(-NaN) = NaN
- Final answer = NaN

### The Fix
Added epsilon smoothing:
```python
entropy = -sum((p + eps) * log(p + eps))
```
Set eps = 1e-10 to avoid log(0) while maintaining accuracy.

Also added output validation:
```python
if isnan(answer) or answer < 0 or answer > 1e6:
    answer = fallback_answer
```

### Lesson
**Numerical stability requires defensive coding.**
- Epsilon smoothing prevents log/divide-by-zero errors
- Output validation catches NaN before submission
- Test entropy computation with sparse distributions

---

## Failure 4: Deterministic Anchor Regression

### The Problem
With temperature=0.0, model produced deterministic but sometimes nonsensical outputs:
- Temperature=0 picks argmax token at each step
- This creates greedy decoding path
- Sometimes stuck in repetition loops (e.g., " ... ... ...")
- Or produced incorrect reasoning chain

### Root Cause
- Base model (GPT-OSS-120B) not fine-tuned for mathematical reasoning
- Temperature=0 reveals weaknesses in base model output distribution
- Exploratory attempts (temp=1.0) sometimes found better paths
- Blind deterministic weighting gave equal credibility to nonsense

### The Fix
Added length-based filtering:
- Skip anchor output if < 50 tokens (likely degenerate)
- Skip if contains >3 repeated n-grams
- Fallback to temperature=0.5 for anchor in these cases

Also adjusted entropy weighting:
- Lower weight for very low-entropy outputs (likely mode collapse)
- Boost weight for diversity in attempt ensemble

### Lesson
**Temperature=0 is not always better, especially with weak base models.**
- Deterministic doesn't mean correct
- Degeneracy detection is crucial
- Ensemble voting should penalize suspicious outputs

---

## Failure 5: Variance Leaderboard Collapse

### The Problem
Score variance was discovered too late:
- Private leaderboard: 34-42 score range across reruns
- Submitted notebook once and scored 41
- Reran for verification: scored 35
- Caused panic about submission validity

### Root Cause
Two sources:

1. **Stochasticity in inference**
   - Temperature=1.0 sampling inherently random
   - Different random seed = different attempts 1-7
   - Variance: ±1.5-2.5 points

2. **Floating-point non-determinism**
   - CUDA reduction operations (sum, argmax) not bit-identical
   - Different GPU memory alignment
   - Rounding in entropy computation
   - Variance: ±0.5-1.0 points

### The Fix
Implemented:
- Seed setting: `torch.manual_seed(42)` + environment variables
- But CUDA operations still non-deterministic
- Solution: Multi-run averaging
  - Run 3 times, average predictions
  - Reduced variance from ±2-3 to ±0.5-1.0 points
  - Increased compute time but improved reliability

Final approach:
- Accepted inherent variance as part of system
- Ensemble across multiple runs when possible
- Documented variance in final report

### Lesson
**Stochasticity is feature, not bug—for this task.**
- Diversity helps find correct answers
- Variance is price of ensemble reasoning
- Bit-determinism expensive (disables GPU optimizations)
- Better to embrace variance through repeated sampling

---

## Summary of Lessons

| Failure | Key Insight | Solution |
|---------|-----------|----------|
| LoRA quantization mismatch | Training precision must match deployment | Retrain with correct precision or abandon fine-tuning |
| BF16 memory explosion | KV cache dominates memory usage | Use quantization, reduce sequence length, or enable offloading |
| Entropy NaN propagation | Numerical instability in probability computation | Add epsilon smoothing and output validation |
| Deterministic anchor regression | Temperature=0 reveals model weaknesses | Add degeneracy detection and adaptive temperature |
| Variance leaderboard collapse | Inference stochasticity unavoidable | Embrace variance, use ensemble averaging, document effects |

## Prevention Checklist

- [ ] Test training precision matches deployment precision
- [ ] Profile memory including KV cache, not just weights
- [ ] Add epsilon (1e-10) to all log/divide operations
- [ ] Validate outputs for NaN/inf/outliers
- [ ] Set random seeds but don't assume bit-determinism
- [ ] Test with multiple runs to measure variance
- [ ] Document stochasticity effects in system
- [ ] Implement fallback/degradation paths for failures
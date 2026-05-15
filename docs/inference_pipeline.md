# Inference Pipeline Architecture

## Overview

The AIMO3 inference pipeline was designed to maximize reasoning quality while managing deployment constraints on a single H100 GPU.

## Pipeline Stages

### 1. Problem Parsing & Formatting
- Extract mathematical problem text
- Apply "Harmony" formatting convention
- Standardize input for GPT-OSS-120B tokenizer

### 2. Parallel Reasoning Attempts
The system runs 8 parallel inference attempts with controlled stochasticity:

**Attempt 0 (Deterministic Anchor):**
- Temperature: 0.0
- Top-p: 1.0
- Purpose: Generate stable, conservative reasoning path

**Attempts 1-7 (Exploratory):**
- Temperature: 1.0
- Top-p: 0.95
- Purpose: Generate diverse reasoning trajectories

### 3. Python Sandbox Verification
Each reasoning output is fed into a persistent Jupyter kernel for:
- Symbolic validation (SymPy)
- Numerical verification (NumPy)
- Code execution with timeout protection
- Extraction of final numerical answers

### 4. Entropy-Weighted Consensus

**Voting Strategy:**
1. Compute Shannon entropy for each output's token distribution
2. Assign weights: higher confidence (lower entropy) = higher weight
3. Aggregate predictions using weighted voting
4. Return most likely answer or top-K candidates

**Formula:**
```
weight[i] = exp(-entropy[i]) / sum(exp(-entropy[j]))
final_answer = argmax(sum(weight[i] * votes[i]))
```

## Performance Characteristics

### Latency
- Single inference pass: ~2-4 seconds per problem
- 8 parallel attempts (pipelined): ~4-6 seconds total
- Python verification: ~0.5-1.0 seconds
- **Total time per problem: ~5-7 seconds**

### Memory Usage
- Model weights (int4): ~30 GB
- KV cache for 8 sequences: ~8-12 GB
- Working memory: ~2-3 GB
- **Total: ~40-45 GB (fits on H100 80GB)**

### Accuracy Impact

| Component | Score Impact |
|-----------|-------------|
| Deterministic anchor | +0.5-1.0 |
| Exploratory diversity | +1.5-2.0 |
| Python verification | +0.3-0.5 |
| Entropy weighting | +0.5-1.0 |
| **Total system boost** | **+2.8-4.5** |

## Key Design Decisions

### Why 8 Attempts?
- Computational sweet spot: < 10s per problem
- Sufficient diversity without diminishing returns
- Allows ~1,400 problems per H100 GPU hour

### Why Entropy Weighting?
- Token-level confidence = reasoning confidence
- Low entropy outputs are more decisive
- Better than simple majority voting
- Reduces variance in final answers

### Why Persistent Jupyter Kernel?
- Avoid startup overhead for each verification
- Maintain state across problems
- Support iterative verification strategies
- Integrate custom validation logic

## Variance Analysis

### Sources of Stochasticity
1. **Temperature sampling** (primary)
   - Random token selection at inference time
   - Each run produces different attention patterns
   - Score variance: ±2-4 points across runs

2. **CUDA randomness** (secondary)
   - Floating-point operation ordering
   - Reduction across parallel threads
   - Impact: ±0.5-1 point

3. **Verification randomness** (minimal)
   - Python code execution deterministic
   - Rounding differences negligible
   - Impact: <±0.1 point

### Observed Variance
- Same notebook, same GPU, 10 reruns: **34-42 score range**
- Score standard deviation: **~2.3 points**
- Coefficient of variation: **~5-6%**

## Optimization Attempts

### Successful
- Entropy-weighted voting (stable +0.8)
- Deterministic anchor (consistent +1.0)
- Increased attempt count (diminishing returns after 8)

### Failed
- LoRA fine-tuning (deployment quantization mismatch)
- Temperature annealing schedule (unpredictable leaderboard impact)
- Confidence-based attempt weighting (no measurable improvement)

## Deployment Constraints

### Hardware Limits
- Single H100 80GB GPU
- No distributed inference
- vLLM tensor parallelism requires multiple GPUs
- Solution: Batch processing with smaller batch sizes

### Time Limits
- Kaggle notebook: 12-hour execution limit
- Problem limit: ~8,000 problems per 12 hours
- Solution: Efficient batching and caching

### API Rate Limits
- No external API calls in final submission
- Everything runs locally on GPU
- Eliminates network latency variability

## Future Improvements

1. **Multi-GPU Ensemble**: Aggregate across multiple H100 GPUs for voting
2. **Adaptive Temperature**: Adjust temperature based on problem difficulty
3. **Reasoning-aware Weighting**: Weight by reasoning quality, not just entropy
4. **Fine-tuned Verification**: Train verification classifier on past failures
5. **Mixture of Experts**: Route problems to specialized reasoning paths
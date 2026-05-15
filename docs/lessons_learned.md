# Lessons Learned from AIMO3

## 1. Stochasticity is a First-Class Citizen

### Key Finding
The same notebook with no code changes produced scores ranging from 34 to 42 across reruns.

### Implications
- Inference variance dominates leaderboard outcomes
- Model quality matters, but not as much as we thought
- Ensemble diversity beats single-model strength
- Bit-determinism is expensive and unnecessary

### Actionable Insight
For reasoning tasks, **embrace stochasticity**:
- Use temperature sampling deliberately
- Design systems for variance (ensemble, voting)
- Test with multiple runs, report confidence intervals
- Avoid false precision (bit-determinism overhead)

---

## 2. Inference Engineering Beats Model Engineering

### Observation
With a fixed base model (GPT-OSS-120B), I achieved more improvement through:
- Deterministic anchor strategy: +1.0 points
- Entropy-weighted voting: +0.8 points
- Python verification: +0.3 points
- **Total system engineering: +2.1 points**

Vs. fine-tuning efforts that failed or yielded <0.5 point gains.

### Why
1. **Base models already powerful**: GPT-OSS-120B is strong (state-of-the-art in 2024)
2. **Fine-tuning trade-offs**: Training cost >> deployment gains
3. **Orchestration multiplies signal**: Multiple attempts + voting > single perfect attempt
4. **Inference is debuggable**: Easy to instrument and optimize at inference time

### Actionable Insight
**Before fine-tuning, max out inference engineering:**
- Ensemble multiple sampling strategies
- Implement domain-specific verification
- Use confidence-weighted aggregation
- Only then consider training-time optimizations

---

## 3. Deployment Constraints Shape Architecture

### The Constraint
Single H100 GPU, 12-hour window, no distributed infrastructure.

### How This Shaped Design
1. **8 parallel attempts**: H100 can handle 8 sequences in KV cache
2. **Deterministic anchor**: "Free" attempt (temperature=0, no sampling randomness)
3. **vLLM over transformers**: vLLM's paging enabled larger batches
4. **Python sandbox**: In-process verification, no API overhead

### Why This Matters
Architecture isn't just about performance—**it's about feasibility**.
- Design space constrained by hardware
- Memory budgets force quantization choices
- Latency budgets force batching strategies
- Each choice cascades to downstream components

### Actionable Insight
**Profile early, design around constraints:**
- Memory: Include KV cache in budgets
- Latency: Include verification time
- Batching: Account for padding and paging
- Test with real deployment hardware, not just local dev machines

---

## 4. Verification is Underrated

### The Opportunity
Mathematical problems have verifiable answers:
- Symbolic validation (SymPy)
- Numerical sanity checks
- Type and range validation
- Consistency with problem constraints

### What We Did
Python sandbox with:
- Safe code execution (timeout protection)
- Multiple validation strategies
- Fallback verification methods
- Runtime answer extraction

### Impact
- +0.3 points in score (modest)
- But caught 2-3 catastrophic failures per submission
- Confidence in submitted answers increased significantly
- Post-competition analysis: verification prevented 5+ negative points

### Actionable Insight
**Verification is a force multiplier:**
- Not just error detection—confidence booster
- Catches failure modes before leaderboard submission
- Enables risk-aware confidence thresholds
- In production systems: verification prevents costly mistakes

---

## 5. AI-Assisted Development is Transformative

### What Happened
Worked with Claude, ChatGPT, Gemini, DeepSeek throughout the project:
- Debugging CUDA errors
- Architecture critique and ideation
- Proof-of-concept implementations
- Deployment troubleshooting

### Time Savings
- 50% faster iteration cycles
- Eliminated some debugging dead-ends
- Rapid prototyping of ensemble strategies
- Error diagnosis in minutes vs hours

### But Not a Panacea
- AI suggested solutions I rejected (LoRA fine-tuning)
- Missed some nuances (KV cache memory profile)
- Required human judgment for final architecture
- Verification and testing still manual

### Actionable Insight
**AI + human judgment > AI alone > human alone:**
- Use AI for rapid exploration and debugging
- Keep humans in loop for architectural decisions
- Verify AI suggestions with domain knowledge
- Most valuable: AI as thinking partner, not oracle

---

## 6. Score Variance Reveals System Dynamics

### The Insight
Large variance (±8 points across runs) wasn't noise—it was signal.

Variance revealed:
1. **Sampling diversity matters**: Different temperatures explore different solution spaces
2. **Ensemble voting works**: Variance in individual attempts reduces in aggregation
3. **Model confidence varies**: Low-entropy outputs converge; high-entropy diverge
4. **Leaderboard is volatile**: Top positions shift with reruns

### Implication for Competition
- Rank 1209 could be anywhere from 1000-1500 with reruns
- "Final score" is misleading—it's a sample, not ground truth
- Ranking determinism requires multiple submissions and averaging

### Implication for ML Systems
- Stochasticity should be managed, not eliminated
- Confidence intervals > point estimates
- Ensemble size should scale with variance
- Reproducibility is aspirational, not achievable with current hardware

### Actionable Insight
**Report uncertainty, not just predictions:**
- Compute confidence intervals from multiple runs
- Use prediction variance to calibrate downstream decisions
- Design systems expecting ~5-10% variance in outputs
- Rank systems by variance-adjusted metrics

---

## 7. Deployment-Aware Optimization Pays Off

### The Mistake
Spent weeks optimizing for training metrics (loss, training accuracy).
Result: Solutions that failed at deployment.

### Examples
1. **LoRA with wrong precision**: Trained well, deployed broken
2. **BF16 model**: Accurate but OOM'd in inference
3. **Long sequence outputs**: Great reasoning, truncated in practice

### The Fix
Adopted deployment-first design:
- Test on actual deployment hardware early
- Measure real latency and memory, not theoretical
- Build degradation paths for constraint violations
- Reserve 20% of compute budget for safety margin

### Result
- No deployment surprises in final submission
- Confident in reliability despite variance
- Clear trade-off decisions (accuracy vs latency)

### Actionable Insight
**Optimize for deployment, not just performance:**
- Deploy early and often
- Profile on real hardware, not local dev setup
- Build fallback and degradation paths
- Reserve resources for unexpected issues
- Test end-to-end, not just components

---

## 8. The Power and Limitation of Ensembles

### What Worked
- 8-attempt ensemble with voting: +2-3 point improvement
- Entropy weighting: +0.5-1.0 point improvement
- Diversity through temperature: key to ensemble effectiveness

### What Didn't
- Fine-tuned models in ensemble: minimal gains, huge complexity
- Different base models: resource-prohibitive on single GPU
- Learned weighting (attention): didn't improve over entropy-based

### Why Simple Works Better
Entropy weighting's elegance:
- Uses model's own confidence (token probabilities)
- No additional training
- Interpretable and debuggable
- Generalizes across problems

### Actionable Insight
**Simple ensembles often beat complex ones:**
- Diversity source matters more than ensemble size
- Temperature sampling is the cheapest diversity
- Entropy weighting: strong baseline, hard to beat
- Learned weighting: not worth complexity unless massive improvement proven

---

## 9. Documentation and Reproducibility

### What I Did Right
- Committed experiment configs to Git
- Logged hyperparameters and seeds
- Saved intermediate outputs
- Detailed notebook annotations

### What I Did Wrong
- Didn't document failure modes early
- Experimental branches got messy
- Post-competition analysis discovered unreported improvements
- No clear "final" version until last moment

### Impact
- Able to reproduce final results within 3-point variance
- Could trace back why certain choices were made
- Post-competition analysis feasible
- But: lost time to version confusion

### Actionable Insight
**Document for your future self:**
- Track experiments in structured format (CSV, not notes)
- Version your models and configurations
- Comment non-obvious hyperparameter choices
- Maintain a "official" branch separate from experiments
- Reproducibility pays off post-competition

---

## 10. Reasoning Systems Need Humility

### The Reality
Even with optimized inference, the system was wrong ~59% of the time (41/69 questions).

### Why
- Base models aren't trained on AIMO data
- Mathematical reasoning is hard
- Single-attempt reasoning insufficient
- No real-time feedback or correction

### Implications
- Ensemble voting helps but doesn't solve fundamental limitation
- Fine-tuning could help (not tried successfully)
- Multi-stage reasoning with verification could help
- Human-in-loop systems would score higher

### Actionable Insight
**Respect the problem difficulty:**
- Don't assume ensemble voting solves all problems
- Measure "ceiling" with upper bounds
- Design for graceful degradation (confidence thresholds)
- In production: escalate uncertain answers to human review
- Plan for system limitations; don't pretend to omniscience

---

## Summary: From Competition to Production

| Competition Finding | Production Application |
|-----------|----------|
| Stochasticity as feature | Design for ensemble and uncertainty |
| Inference engineering > model engineering | Optimize orchestration before training |
| Deployment shapes architecture | Profile on real hardware from day 1 |
| Verification is underrated | Build validation into every pipeline |
| AI as thinking partner | Use AI for exploration, humans for judgment |
| Manage variance explicitly | Report confidence intervals, not point estimates |
| Deploy-aware optimization | Test full pipelines early |
| Simple ensembles work best | Resist complexity; favor diversity |
| Document everything | Track experiments, versions, decisions |
| System has limits | Design graceful degradation and escalation |

---

## Future Work

1. **Fine-tune with correct deployment setup** (failed once, worth retry)
2. **Multi-GPU ensemble** (aggregate across multiple H100s)
3. **Adaptive temperature** (per-problem difficulty routing)
4. **Verify with trained scorer** (learn to rate reasoning quality)
5. **Hybrid symbolic-neural** (combine with SymPy engine)
6. **Chain-of-thought distillation** (smaller model with better reasoning)

The core insight remains: **reasoning systems benefit from thoughtful engineering, not just model scaling.**
# Inside AIMO3
## Engineering Stable Reasoning Under Stochastic Inference

This repository documents my work for Kaggle's AI Mathematical Olympiad (AIMO3).

**Final Result:**
- Private Score: 41.0
- Rank: 1209 / 4,138 teams
- Hardware: Single H100 GPU
- Top Score: 44.0

## Overview

This project explored:

- GPT-OSS-120B inference engineering
- vLLM deployment
- entropy-weighted consensus voting
- deterministic anchor sampling
- Python sandbox verification
- stochastic reasoning variance
- LoRA fine-tuning attempts
- deployment-aware optimization

## Key Insights

One of the most important findings from AIMO3 was the impact of stochastic variance.

The exact same notebook produced scores ranging from 34 to 42 across reruns.

This competition demonstrated that leaderboard outcomes were influenced not only by model quality, but also by inference stability and reasoning trajectory variance.

## Core Architecture

```
Problem Input
    ↓
Harmony Formatting
    ↓
GPT-OSS-120B
    ↓
8 Parallel Reasoning Attempts
    ↓
Python Sandbox Verification
    ↓
Entropy-Weighted Consensus
    ↓
Final Numerical Answer
```

## Technical Components

### Deterministic Anchor Strategy
- Attempt 0: Temperature = 0.0
- Attempts 1-7: Temperature = 1.0

This balanced deterministic stability with exploratory reasoning diversity.

### Entropy-Weighted Voting
Outputs with lower entropy received higher aggregation weight during final answer selection.

### Python Verification Layer
Integrated symbolic and numerical validation using:
- SymPy
- NumPy
- brute-force verification
- persistent Jupyter kernels

## Fine-Tuning Attempts

LoRA fine-tuning succeeded during training but failed during deployment because of:

- MXFP4 quantization mismatch
- BF16 memory explosion
- H100 VRAM constraints

This highlighted the importance of deployment-aware engineering.

## AI-Assisted Workflow

This project was heavily AI-assisted.

I worked with:
- ChatGPT
- Claude
- Gemini
- Qwen
- DeepSeek

for:
- architecture critique
- debugging
- reasoning discussions
- optimization strategies
- deployment troubleshooting

## Documentation

- **Technical Report:** `report/AIMO3_REPORT_VISUAL_FINAL.pdf`
- **Slides:** `slides/AIMO3_Technical_Report.pptx`
- **Notebooks:** See `notebooks/` directory
- **Detailed Docs:** See `docs/` directory

## Directory Structure

```
aimo3-inference-engineering/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── report/
│   ├── AIMO3_REPORT_VISUAL_FINAL.pdf
│   └── images/
│
├── notebooks/
│   ├── final_submission.ipynb
│   ├── deterministic_anchor_experiment.ipynb
│   └── variance_analysis.ipynb
│
├── slides/
│   └── AIMO3_Technical_Report.pptx
│
├── assets/
│   ├── architecture.png
│   ├── variance_chart.png
│   ├── leaderboard.png
│   └── fine_tune_wall.png
│
├── experiments/
│   ├── submission_history.csv
│   ├── score_tracking.csv
│   └── notes.md
│
└── docs/
    ├── inference_pipeline.md
    ├── deployment_failures.md
    └── lessons_learned.md
```

## Disclaimer

This repository is intended for research, learning, and documentation purposes.

Some notebook components are simplified for readability.
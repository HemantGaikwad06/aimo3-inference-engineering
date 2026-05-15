# Repository Structure Guide

```
aimo3-inference-engineering/
│
├── README.md                          # Project overview, key results, architecture
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── .github/
│   ├── CONTRIBUTING.md               # Contribution guidelines
│   └── README_STRUCTURE.md           # This file
│
├── docs/
│   ├── inference_pipeline.md         # Detailed architecture, memory/latency profiles
│   ├── deployment_failures.md        # 5 major failures + root cause analysis
│   └── lessons_learned.md            # 10 comprehensive insights
│
├── notebooks/
│   ├── final_submission_architecture.md  # V54 implementation details
│   ├── final_submission.ipynb           # (TODO: Add actual notebook)
│   ├── deterministic_anchor_experiment.ipynb  # (TODO: Ablation study)
│   └── variance_analysis.ipynb          # (TODO: Variance analysis)
│
├── report/
│   ├── AIMO3_REPORT_VISUAL_FINAL.pdf    # (TODO: Upload visual report)
│   └── images/
│       ├── architecture.png             # (TODO: System architecture diagram)
│       ├── variance_chart.png           # (TODO: Score distribution)
│       ├── leaderboard.png              # (TODO: Final leaderboard)
│       └── fine_tune_wall.png           # (TODO: Fine-tuning failure analysis)
│
├── slides/
│   └── AIMO3_Technical_Report.pptx      # (TODO: Technical presentation)
│
├── assets/
│   ├── architecture_diagram.txt      # ASCII architecture
│   ├── performance_metrics.csv       # System performance data
│   └── component_contributions.csv   # Contribution of each component
│
└── experiments/
    ├── submission_history.csv        # (TODO: Version history with scores)
    ├── score_tracking.csv            # (TODO: Detailed scoring breakdown)
    └── notes.md                      # (TODO: Experimental notes)
```

## File Descriptions

### Root Files
- **README.md**: Start here. Contains overview, final results, architecture diagram, and key insights.
- **requirements.txt**: All Python dependencies needed to reproduce or extend the work.
- **LICENSE**: MIT license for open-source usage.

### docs/ (Core Documentation)
- **inference_pipeline.md**: Comprehensive pipeline design with performance analysis.
- **deployment_failures.md**: Chronicles 5 major failures with root cause analysis and lessons.
- **lessons_learned.md**: 10 high-level insights applicable to production systems.

### notebooks/ (Implementation Details)
- **final_submission_architecture.md**: Technical breakdown of V54 (submitted code).
- **final_submission.ipynb**: (Optional) Standalone notebook demonstrating the approach.
- **deterministic_anchor_experiment.ipynb**: Ablation study showing temperature=0 benefit.
- **variance_analysis.ipynb**: Analysis of score variance across runs.

### report/ (Visual Documentation)
- **AIMO3_REPORT_VISUAL_FINAL.pdf**: Comprehensive visual report with charts and insights.
- **images/**: Supporting visuals (architecture, variance, leaderboard, failure analysis).

### slides/ (Presentations)
- **AIMO3_Technical_Report.pptx**: Technical presentation for conferences/discussions.

### assets/ (Data & Diagrams)
- **architecture_diagram.txt**: ASCII representation of system architecture.
- **performance_metrics.csv**: Raw performance data (latency, memory, throughput).
- **component_contributions.csv**: Score contribution from each system component.

### experiments/ (Tracking)
- **submission_history.csv**: Version history with scores and key changes.
- **score_tracking.csv**: Detailed scoring breakdown by problem category.
- **notes.md**: Experimental notes and observations.

## How to Navigate

### For a Quick Overview
1. Read `README.md` (5 min)
2. Skim `docs/lessons_learned.md` (10 min)

### For Technical Deep-Dive
1. Read `docs/inference_pipeline.md` (15 min)
2. Read `notebooks/final_submission_architecture.md` (20 min)
3. Review `docs/deployment_failures.md` (15 min)

### For Reproduction/Extension
1. Install `requirements.txt`
2. Review `notebooks/final_submission_architecture.md`
3. Reference `docs/inference_pipeline.md` for system parameters
4. Check `docs/deployment_failures.md` for common pitfalls

### For Production Deployment
1. Read `docs/lessons_learned.md` (production checklist)
2. Review `docs/deployment_failures.md` (failure modes)
3. Reference performance profiles in `docs/inference_pipeline.md`
4. Check `experiments/notes.md` for configuration tuning

## Key Metrics

| Metric | Value | Location |
|--------|-------|----------|
| Final Score | 41.0 / 69 | README.md |
| Rank | 1209 / 4138 | README.md |
| Score Variance | ±2-3 points | notebooks/final_submission_architecture.md |
| Memory Usage | 45 GB | docs/inference_pipeline.md |
| Latency/Problem | 5-7 seconds | docs/inference_pipeline.md |
| Throughput | 6000-8000 problems/12h | docs/inference_pipeline.md |

## Contributing

See `.github/CONTRIBUTING.md` for contribution guidelines.

# AI Portfolio (2025–2029) — Perception + Production + GenAI

This repository is my **portfolio focused on “usable AI systems”**: measured, reproductible models that can be deployed (API, Docker, monitoring).
Goal: become a **Machine Learning Engineer specialized in Computer Vision / Perception**.


## Roadmap 
- **2025 — ML Foundations**: Titanic (classification) + House Prices (regression) + mini MLOps
- **2026 — Deep Learning & Industrial Vision**: CNN (MNIST/CIFAR-10) + Smart Inspection System
- **2027 — Vision–Language & GenAI**: Semantic Search (CLIP) + VLM scene analysis
- **2028 — Edge AI & Optimization**: ONNX + quantization + latency vs accuracy benchmark
- **2029 — Robotics Capstone**: perception + multimodal + real-time (collaborative robot)

## Vision
- **Reproducibility**: fixed seeds, CV, clean configs.
- **Measurement**: clear metrics, reports, and error analysis.
- **Deployment**: `/predict` + `/health` FastAPI (/predict, /health) + Docker.
- **Quality**: clean project structure, tests, linting, CI.
- **Artifacts**: metrics.json, report.md, figures/


## Repository structure

- `01_ML_Foundations/` : classical ML projects (Scikit-Learn pipelines)
- `02_DL_Vision_Industrielle/` : deep learning + vision (product-style projects)
- `03_Vision_Language_GenAI/` : CLIP, embeddings, vector DB, VLM
- `04_Edge_AI_Optimisation/` : ONNX export, quantization, benchmarking
- `05_Capstone_Robotique/` : final robotics/perception project
- `docs/` : global documentation (roadmap, conventions, guide)
- `templates/` : reusable project skeletons (API, tests, CI, Docker)
- `tools/` : utility scripts (lint all, tests all, init project…)


## How to run a project
Each project is self-contained and has its own README.
Typical steps:
1. Create an environment (`.venv`) and install dependencies
2. Run scripts 
3. Review reports (metrics, figures, error analysis)
4. (if available) run the API (FastAPI) and test `/health` + `/predict`


## Quality criteria (what each project should include)
- Metrics + model comparisons
- Reproducibility (seeds, configs)
- Feature pipeline (when relevant)
- Reporting (JSON/Markdown) + plots
- Packaging + CI (for “production” projects)

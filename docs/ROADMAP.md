# Roadmap 2025–2029 — Perception + Production + GenAI

Goal: build a portfolio that demonstrates **usable AI systems** (measured, reproducible, deployable).

## 2025 — ML Foundations
### Titanic — Classification
- Reproductible pipeline (baseline → improvements)
- Proper validation (CV, stratified split)
- Metrics: Precision/Recall/F1 (+ ROC-AUC if relevant)
- Deliverables: report + model comparison + README

### House Prices — Regression
- Pipeline + reasonable feature engineering
- Metrics: MAE/RMSE
- Error analysis (residuals, segments, outliers)
- Deliverables: report + README

### Mini MLOps (applied to House Prices)
- Run tracking (MLflow)
- CI (lint + tests)
- Docker + FastAPI API (`/predict`, `/health`)
- Versioned artifacts (metrics/plots) with clean structure

## 2026 — Deep Learning & Industrial Vision
### CNN MNIST
- CNN basics, stable training, overfit vs augmentation
- Deliverables: notebook, curves, mini-report

### CNN CIFAR-10
- More complex images + transfer learning
- Architecture comparison + results/limitations

### Smart Inspection System (product project)
- FastAPI + Streamlit + Docker
- Monitoring KPIs (Precision/Recall/FPS)
- PDF report export (scores + history)

## 2027 — Vision–Language & GenAI
### Semantic Search (CLIP)
- Image↔Text, embeddings, vector DB
- Recall@K evaluation

### VLM Scene Analysis
- Explanations and “why rejected”
- Interactive demo

## 2028 — Edge AI & Optimization
- ONNX export
- FP16/INT8 quantization
- Latency/FPS profiling
- Real-time webcam demo

## 2029 — Robotics Capstone
- Collaborative robot: perception + multimodal + edge + real-time
- Demo video + architecture docs + Docker Compose + stability benchmarks

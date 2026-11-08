---
title: VisionGuard Model API (21L-6193_A2_Ops)
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
---

# VisionGuard model API (MLOps Assignment 2, Task 7)

AI-generated vs real image classifier (MobileNetV2 champion from the MLflow registry), served
by `api/server.py` with gunicorn. Source: https://github.com/taha-nu/21L-6193_A2_Ops

- `GET  /api/v1/health`  public: model name, version and threshold
- `POST /api/v1/predict` multipart field `image`, header `X-API-Key: <API_KEY>`

`API_KEY` is a Space secret (Settings > Variables and secrets), never stored in these files.

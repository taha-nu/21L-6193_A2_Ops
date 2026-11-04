# VisionGuard — AI vs Real Image Classification (`21L-6193_A2_Ops`)

![CI](https://github.com/taha-nu/21L-6193_A2_Ops/actions/workflows/ci.yml/badge.svg)

Upload an image, get **AI-generated / Real**, a confidence value and an honest caveat.
TensorFlow/Keras model · MLflow on DagsHub · DVC · Flask · PostgreSQL · Docker Compose · GitHub Actions · cloud model API.

| Link | |
|---|---|
| GitHub | `https://github.com/taha-nu/21L-6193_A2_Ops` |
| DagsHub (code, data, MLflow) | `https://dagshub.com/taha-nu/21L-6193_A2_Ops` |
| Model API | `http://<vm-ip>:8080/api/v1/health` |

> Naming rule: `21L-6193_A2_Ops` for the GitHub repo, DagsHub repo, MLflow experiment (`ROLL_NO` in `.env`) and the cloud instance. Replace `taha-nu` with your DagsHub username once the account exists.

## Repository map
```
src/            audit.py · eda.py · prepare.py · data.py (tf.data) · models.py · train.py (MLflow)
                evaluate.py · error_analysis.py · gradcam.py · tune.py · registry.py · common.py
visionguard/    preprocess.py (shared train/serve) · inference.py (Predictor) · validation.py
app/            Flask UI: routes, DB model, remote API client, templates, static
api/            model API for the cloud (server.py)
scripts/        run_experiments.sh · fetch_model.py · load_champion.py · Postman collection
labs/           toy parts: env_check · dvc_toy.sh · mlflow_toy · tuner_toy · docker_warmup.sh · ci_toy/
docker/         Dockerfile (web, multi-stage) · Dockerfile.naive (size baseline) · Dockerfile.api
docs/           DEPLOY.md (Tasks 7–8) · OPERATIONS.md (Task 6.3–6.5)
report/         REPORT_DRAFT.md — chapter skeleton with explanations, fill in your numbers/screens
params.yaml     all default hyper-parameters
```

## Run order (what to do, in order)

### 0. Setup (Task 1.1)
```bash
git clone <this repo> && cd 21L-6193_A2_Ops
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                        # fill in ROLL_NO + DagsHub values
python labs/env_check.py                                    # 📸 versions + GPU check
```

### 1. DagsHub repo + data with DVC (Tasks 1.4, 3.2)
1. On DagsHub: *Create → New Repository → Connect a repository → GitHub*, pick `21L-6193_A2_Ops`. Make it public.
2. *Your settings → Tokens*: copy your token. Repo page → *Remote → Experiments* shows the MLflow URI
   `https://dagshub.com/<user>/21L-6193_A2_Ops.mlflow`. Put both in `.env` (never in code). `.env` is in `.gitignore`.
3. Toy DVC lab: `bash labs/dvc_toy.sh` (run block by block, screenshot each).
4. Download the Kaggle dataset (`kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images`)
   and unzip into `data/raw/` (keep its `AiArtData/` and `RealArt/` folders). Then:
```bash
dvc init
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/21L-6193_A2_Ops.s3
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc/config && git commit -m "track raw dataset with DVC"
dvc push && git push
```

### 2. Data understanding (Tasks 1.2, 1.3)
```bash
python -m src.audit      # reports/audit.csv + audit_summary.md (counts, min/median/max, corrupt, duplicates)
python -m src.eda        # reports/figures/sample_grid.png, hist_width/height/file_size/aspect.png
python -m src.prepare    # data/processed/ (clean RGB JPEGs) + data/splits/{train,val,test}.csv
dvc add data/processed && git add data/processed.dvc data/splits && git commit -m "processed data v1" && dvc push
python -m src.data       # reports/figures/augmented_batch.png  (Task 2.1)
```

### 3. Experiments (Tasks 2, 3)
```bash
python labs/mlflow_toy.py     # Task 3.1: unset MLFLOW_TRACKING_URI first for the local run, then `mlflow ui`
python labs/mlflow_toy.py     # Task 3.2: again with DagsHub variables set
python labs/tuner_toy.py      # Task 2.7 toy
bash scripts/run_experiments.sh   # 9 runs + comparison + registry (2 versions, @champion) + fetch model
python -m src.tune --trials 10    # Task 2.7 apply
python -m src.error_analysis --run-id <best run id>   # Task 2.6
python -m src.gradcam --run-id <best run id>          # bonus
python scripts/load_champion.py some_image.jpg        # Task 3.6
```
If CPU training is slow, add `--img-size 160` or reduce `epochs` in `params.yaml`; or run the same
commands in Google Colab with a GPU (clone repo, `dvc pull`, set the env vars).

### 4. Web app (Tasks 4, 5)
```bash
python -m visionguard.inference real.jpg ai.png   # Task 4.1 console output
python run.py                                      # http://127.0.0.1:5000  (debug off, SQLite)
pytest -v                                          # tests incl. the 3 failure cases
```

### 5. Docker (Task 6) — see `docs/OPERATIONS.md`
```bash
bash labs/docker_warmup.sh
docker compose up -d --build && docker compose ps     # http://localhost:8000
```

### 6. Cloud API + integration (Tasks 7, 8) — see `docs/DEPLOY.md`

## Configuration (environment variables)
| Variable | Used by | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | training | DagsHub MLflow server + token |
| `ROLL_NO` / `PROJECT_NAME` | training | MLflow experiment name `21L-6193_A2_Ops` |
| `DATABASE_URL` | web | `postgresql+psycopg://…@db:5432/…` (default SQLite) |
| `INFERENCE_MODE` | web | `local` model, `remote` cloud API, `mock` for tests |
| `MODEL_PATH`, `THRESHOLD` | web, api | model file; threshold override (default from `serving_config.json`) |
| `API_URL`, `API_KEY` | web (remote), api | cloud API address and shared secret |
| `SECRET_KEY`, `MAX_UPLOAD_MB` | web | Flask signing key, upload limit |

## CI
`.github/workflows/ci.yml` runs `ruff`, `pytest` (mock model, SQLite) and builds the Docker image on
every push. Secrets for workflows go in *GitHub → Settings → Secrets and variables → Actions* and are
used as `${{ secrets.NAME }}`; they never appear in the YAML because the repository (and its history)
is readable by anyone with access, and GitHub masks secret values in logs.

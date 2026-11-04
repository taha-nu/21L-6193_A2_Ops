# 21L-6193_VisionGuard — MLOps Assignment 2

> **How to use this draft.** It follows the required structure (one chapter per task, one section per
> sub-task ID). The explanations describe what the code in this repo does and why. Everything in
> `[FILL: …]` needs your numbers, and every `📸` needs your screenshot (terminal prompt with your
> username + date, or browser URL bar). Rewrite the text in your own words and add "what went wrong"
> notes from your actual runs. Then export to PDF as `21L-6193_VisionGuard.pdf`.

## Front matter
- **Name:** [FILL] **Roll number:** [FILL]
- **GitHub:** https://github.com/[FILL]/21L-6193_A2_Ops
- **DagsHub:** https://dagshub.com/[FILL]/21L-6193_A2_Ops
- **Live model API:** [FILL: http://<ip>:8080] **Graded commit:** [FILL: `git rev-parse HEAD`]
- **AI assistance / tutorials disclosure:** [FILL honestly, e.g. "Claude (Anthropic) was used to scaffold
  the code structure, Dockerfiles and a draft of explanations; I ran all experiments, deployment and
  screenshots myself and edited the text." + any tutorials you followed.]

---

## Chapter 1 — Data Understanding and Versioning

### 1.1 Environment
Created an isolated virtual environment (`python3.12 -m venv .venv`) so project packages do not clash
with system Python, and pinned versions in `requirements*.txt` for reproducibility. `labs/env_check.py`
prints Python, TensorFlow, Keras, MLflow, DVC versions and lists GPUs visible to TensorFlow.
📸 Fig 1.1 — output of `python labs/env_check.py`. GPU: [FILL: none → CPU training / GPU name].
TensorFlow-CPU 2.21 was chosen because [FILL: no NVIDIA GPU]; the CPU wheel is much smaller.

### 1.2 Dataset audit
`src/audit.py` walks `data/raw`, and for every file records class (from the parent folder name),
file size, width, height, colour mode, format and SHA-256 of the bytes. It opens every image with
Pillow **and calls `load()`**, which forces a full decode; `open()` alone only reads the header and
misses truncated files. Exact duplicates = identical SHA-256; we also check whether the same bytes
appear under both classes (a label conflict).

Table 1.1 — counts per class: [FILL from reports/audit_summary.md]
Table 1.2 — min / median / max of width, height, file size per class: [FILL]
Corrupt files: [FILL] Duplicates: [FILL] Cross-class duplicates: [FILL]
Interpretation: [FILL, e.g. AI images cluster at a few fixed generator resolutions (512, 768, 1024) while
real images vary widely; real images are larger files. This is a shortcut risk: a model could learn
"size/format" instead of "AI-ness", which is one reason all images are re-encoded to the same JPEG
quality and resized to 224×224 before training.]

### 1.3 Visual EDA
📸 Fig 1.2 sample grid — [2–3 sentences: subjects, style differences, any watermarks / paintings in "real"].
📸 Fig 1.3 width histogram — [2–3 sentences]. 📸 Fig 1.4 height — [..]. 📸 Fig 1.5 file size (log scale) — [..].
📸 Fig 1.6 aspect ratio — [..; motivates resize_with_pad in 2.1].

### 1.4 DVC
**Toy:** `labs/dvc_toy.sh` — `git init`, `dvc init`, `dvc add data.csv`, commit v1, edit, commit v2,
`git checkout HEAD~1 -- data.csv.dvc && dvc checkout` returns the 3-row version; DagsHub storage
configured as an S3-compatible remote (credentials stored with `--local` in the git-ignored
`.dvc/config.local`); `dvc push`, delete data and cache, `dvc pull` restores it. 📸 each stage.
**Apply:** `dvc add data/raw` and `dvc add data/processed`, pushed to DagsHub storage. 📸 DagsHub
data tab.
**What the pointer file holds:** a few lines of YAML: the MD5 hash of the data (for a directory, the
hash of a manifest listing every file), its size, number of files and path. Git versions this tiny file;
the actual bytes live in the DVC cache/remote, addressed by that hash. Checking out an old commit
brings back the old hash and `dvc checkout` fetches the matching data.
**Why large data does not belong in Git:** Git stores every version of every file forever in its history
and every clone downloads all of it; binary images do not diff, so each change adds a full copy; hosting
providers limit file and repo size (GitHub: 100 MB per file). DVC keeps Git small and still gives
exact data versions (the `dvc_data_version` tag on every MLflow run is this hash).

---

## Chapter 2 — Model Development

### 2.1 Input pipeline
Before tf.data, `src/prepare.py` drops corrupt files, duplicates and conflicts, converts every image to
RGB (palette/RGBA/greyscale images flattened; transparency on white) and re-encodes it as JPEG
quality 95, then makes a stratified 70/15/15 split with seed 42 (Table: [FILL counts]). De-duplicating
**before** splitting prevents the same picture appearing in train and test.
`src/data.py` stages: `from_tensor_slices(paths, labels)` → `shuffle` (train only, reshuffled each epoch)
→ `map(read_file → decode_jpeg → resize_with_pad)` in parallel → `cache` (val/test) → `batch(32)` →
`map(augment)` (train only: horizontal flip, rotation ±18°, zoom ±10 %, contrast ±10 %) → `prefetch`.
**Image size 224×224:** the native input size of MobileNetV2/EfficientNetB0's ImageNet weights, and
small enough for CPU training. **Aspect ratio:** `resize_with_pad` scales the long side to 224 and pads
the short side with black, so nothing is stretched (stretching distorts textures and faces, the very cues
that reveal generation). Augmentation is train-only because validation/test must measure the model on
unmodified images. The same `visionguard/preprocess.py` is used by the server (see 4.1).
📸 Fig 2.1 one augmented batch.

### 2.2 Baseline CNN
`src/models.py::build_baseline`: Rescaling(1/255) → 4 × [Conv2D 3×3 (32/64/128/256) → BatchNorm →
ReLU → MaxPool] → GlobalAveragePooling → Dropout(0.4) → Dense(1, sigmoid).
Why each layer: convolutions learn local patterns (edges, textures, upsampling artefacts); batch
normalisation keeps activations in a stable range so training is faster and less sensitive to the learning
rate; ReLU adds non-linearity; max-pooling halves resolution and widens the receptive field; global
average pooling turns each feature map into one number (far fewer parameters than Flatten → less
overfitting on ~1k images); dropout randomly removes 40 % of features during training as regularisation;
one sigmoid unit outputs P(AI). **Loss:** binary cross-entropy — the negative log-likelihood of a
Bernoulli label, the natural pair for one sigmoid. **Optimizer:** Adam (lr 1e-3), adaptive per-parameter
steps, robust default. **Metrics:** accuracy plus precision, recall and AUC, because accuracy hides which
error the model makes.
📸 Fig 2.2 `model.summary()`. Table 2.1 — layer / output shape / params: [paste reports/layer_table.md
artifact from the baseline run].

### 2.3 Transfer learning
MobileNetV2 (ImageNet weights, `include_top=False`) chosen because it is small (~2.3 M backbone
params, ~10 MB) and fast on CPU — important for serving on a free VM. **Stage 1:** backbone frozen; only
the head (GAP → Dropout 0.3 → Dense sigmoid) trains, lr 1e-3. **Stage 2:** unfreeze only the last block
(`block_16_*`, `Conv_1`), BatchNorm layers kept frozen, recompile with lr 1e-5 (100× smaller) so the
pretrained filters are adjusted gently rather than destroyed.
**Preprocessing:** MobileNetV2 was trained on pixels scaled to [-1, 1] (`x/127.5 − 1`); feeding [0, 255]
or [0, 1] would put the inputs in a range the pretrained filters never saw. We implement it as a
`Rescaling(1/127.5, offset=-1)` layer **inside** the model, so the saved model carries its own
preprocessing. The backbone is called with `training=False` so BatchNorm uses ImageNet statistics.
📸 Fig 2.3 stage 1 curves, Fig 2.4 stage 2 curves (artifacts `curves_stage1_frozen.png`,
`curves_stage2_finetune.png`). Trainable layers: stage 1 → artifact `trainable_stage1.txt` (head only);
stage 2 → `trainable_stage2.txt` [FILL: N layers].

### 2.4 Training discipline
Seeds fixed with `keras.utils.set_random_seed(42)` (Python, NumPy, TF) plus seeded shuffle/augment.
Callbacks: EarlyStopping (val_loss, patience 5, restore best weights), ModelCheckpoint (save only when
val_loss improves; stage 2 starts from stage 1's best value so it only overwrites with a genuinely better
model), ReduceLROnPlateau (×0.3 after 2 flat epochs). Final model = checkpoint with the lowest val_loss.
Best epoch: baseline [FILL], MobileNetV2 [FILL]. 📸 Fig 2.5 curves with the best epoch marked.
Note: full bit-for-bit determinism on GPU would also need `tf.config.experimental.enable_op_determinism()`.

### 2.5 Evaluation
Positive class = AI-generated. **Which error is worse?** Calling a real photo fake (false positive): it
amounts to accusing a person of fabrication — reputational harm to photographers, journalists, students.
A missed fake (false negative) is bad but the tool is only one signal. So the threshold is chosen on the
**validation** set as the lowest value whose false-positive rate is ≤ 5 % (maximising recall under that
cap), then the test set is evaluated **once** with that fixed threshold.
Table 2.2 (test set): | model | threshold | accuracy | precision | recall | F1 | ROC-AUC | FPR | [FILL from
`test_*` metrics of baseline-cnn and mnv2-ref].
📸 Fig 2.6/2.7 confusion matrices, Fig 2.8/2.9 ROC curves (red dot = chosen threshold).
[FILL: 3–4 sentences comparing models.]

### 2.6 Error analysis
`python -m src.error_analysis --run-id <id>` shows the 10 most confident mistakes with size, aspect,
sharpness (variance of edges; low = blurry) and saturation. 📸 Fig 2.10.
Table 2.3 hypotheses: [FILL one line per image, e.g. "real, painting-like illustration → model associates
smooth shading with AI"; "AI, heavily JPEG-compressed thumbnail 256 px → generator artefacts erased";
"real, studio portrait with bokeh → smooth background looks synthetic"; "AI with watermark text"].

### 2.7 Keras Tuner
**Toy:** `labs/tuner_toy.py`, Hyperband on Fashion-MNIST over units (32–512) and lr (1e-4–1e-2).
Table 2.4 [FILL top trials, best settings, test accuracy].
**Apply:** `python -m src.tune --trials 10`, RandomSearch over lr (1e-5–3e-3 log), dropout (0–0.6) and
unfrozen layers (0/10/20/40/60). Table 2.5 [FILL reports/tuner_results.csv]. Most important setting:
[FILL: highest |Spearman| with val_loss, usually learning rate — too high destroys pretrained features
when layers are unfrozen, too low barely moves the head]. **Why the best validation score overfits the
validation set:** with many trials, some score well partly by luck on those specific ~150 images;
selecting the maximum picks that luck, so the best val score is optimistically biased. The untouched test
set (used once) gives the honest estimate; with more data, nested CV or a second holdout helps.

**Bonus — Grad-CAM:** 📸 Fig 2.11 (`src/gradcam.py`). [FILL: what regions light up — faces/hands,
backgrounds, textures, or borders/padding (a shortcut warning sign).]

---

## Chapter 3 — Experiment Tracking

### 3.1 MLflow locally
`labs/mlflow_toy.py` trains a Dense network on Fashion-MNIST twice (lr 1e-3, 1e-2), logging params,
per-epoch metrics, a loss plot and the model; `mlflow ui` → compare. 📸 both runs + comparison view.
Definitions (in my words): **experiment** — a named folder grouping runs that attack the same problem;
**run** — one execution of training code with its own ID, start time and status; **parameter** — an input
setting fixed before the run (lr, batch size), logged once; **metric** — a numeric result, possibly logged
at many steps (val_loss per epoch); **artifact** — any output file attached to a run (plots, model, CSVs);
**tag** — free-form key/value metadata used to describe and filter runs (git commit, author, data version).

### 3.2 DagsHub
DagsHub repo connected to GitHub. Tracking URI + token given to code through env variables
`MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD` loaded from `.env` by python-dotenv; `src/common.py`
never contains credentials. 📸 DagsHub project page, toy run on DagsHub, and proof `.env` is ignored:
`git check-ignore -v .env` → `.gitignore:2:.env  .env`, and `git status` not listing it.

### 3.3 Instrumenting VisionGuard
Both modes add tags `git_commit`, `dvc_data_version` (MD5 from the `.dvc` pointer), `seed`, model type;
log all hyper-parameters and final `test_*` metrics.
- **Manual** (`--tracking manual`, `MLflowEpochLogger` callback): logs exactly train/val loss, accuracy,
  precision, recall, AUC and current lr each epoch, with a step counter that continues across stage 1 and
  stage 2, so one continuous curve.
- **Autolog** (`mlflow.tensorflow.autolog()`): automatically logs optimizer name/params, epochs,
  batch size, early-stopping settings, per-epoch metrics and the model summary with no code. It misses
  our test metrics, threshold, data version and git tags (added by hand), restarts its step counter on the
  second `fit()` (two overlapping curves), and its model logging saves the model at the end of each fit
  rather than the best checkpoint across both stages, so we disabled it and log the model ourselves.
📸 run page params / metrics / tags for `mnv2-ref` and `mnv2-autolog`.

### 3.4 Artifacts
Each run attaches `reports/` (confusion_matrix.png, roc_curve.png, training curves, test_predictions.csv,
test_metrics.json, serving_config.json, layer_table.md, model_summary.txt, `requirements-freeze.txt` =
package list), `splits/` (train/val/test CSVs) and the model (`keras_model/model.keras` + MLflow `model/`).
📸 Artifacts tab.

### 3.5 Comparison
`scripts/run_experiments.sh` → runs differing in one change from `mnv2-ref`: lr 3e-4, no augmentation,
no fine-tuning, 40 unfrozen layers, 160 px, EfficientNetB0 (+ baseline, autolog twin).
`python -m src.registry compare` → Table 3.1 / Fig 3.1. 📸 DagsHub compare view.
**Choice:** [FILL best run]. Primary metric ROC-AUC (threshold-independent ranking quality) and recall at
the FPR ≤ 5 % operating point (matches the product requirement); F1 as tie-breaker; then model size /
latency for serving. [FILL justification with numbers.]

### 3.6 Registry
`register-best --top 2` registers the two best runs as versions of `VisionGuard`; the best gets alias
**champion** (the other **challenger**). `scripts/load_champion.py img.jpg` loads `models:/VisionGuard@champion`
and prints the prediction. Aliases decouple "which model is in production" from code: re-pointing the
alias promotes a new version without editing anything.
📸 registry page with both versions + alias; terminal output of load_champion.
**How the champion reaches the server:** `scripts/fetch_model.py` downloads it **once** at deploy/build
time into `models/` (with `serving_config.json`: threshold, image size, version). The web container mounts
it; the API image copies it in. No request ever calls DagsHub.

---

## Chapter 4 — Flask Web Application

### 4.1 Inference module
`visionguard/inference.py::Predictor` loads the `.keras` file once when the app is created, reads the
threshold from `serving_config.json` and runs a warm-up prediction. **Why not per request:** loading
takes seconds and hundreds of MB; per request it would make each response slow, multiply memory use
under concurrent requests and could crash the container. **Why preprocessing must match:** the model
only learned the input distribution it was trained on; a different resize (stretch vs pad), colour order,
scaling or JPEG step shifts that distribution (training/serving skew) and silently lowers accuracy with no
error message. We share one implementation, and `tests/test_preprocess.py` asserts train and serve
tensors are identical. 📸 console: `python -m visionguard.inference real.jpg ai.png`.
**Confidence** is shown relative to the threshold (0.5 = on the threshold, 1.0 = maximally far), next to
the raw P(AI), so a "Real" verdict never shows "<50 % confident".

### 4.2 Routes and validation
Routes: `/` upload, `POST /predict` result, `/history`, `/health` (JSON), `/api/history` (JSON).
`visionguard/validation.py`: (1) extension allow-list → 415; (2) size: Flask `MAX_CONTENT_LENGTH`
rejects large bodies before reading them, plus an explicit check → 413; (3) full Pillow decode + format
check, and a pixel cap against decompression bombs → 422. `secure_filename` strips paths and odd
characters (`../../etc/my photo.png` → `etc_my_photo.png`). The upload is read into memory only and
never written to disk; only its SHA-256 is stored. Friendly `error.html` page for every error.
📸 three failure screenshots: .pdf/.txt upload, >8 MB file, `.jpg` that is really text.

### 4.3 HTML/CSS
Jinja templates `base/index/result/history/error.html` + `static/style.css`: drag-and-drop upload with
preview, verdict card with confidence bar, model version, threshold and caveat; History in the top menu
reads from the database. 📸 upload page, result page, history page.

### 4.4 Configuration and server
All settings from environment variables (`DATABASE_URL`, `MODEL_PATH`, `THRESHOLD`, `INFERENCE_MODE`,
`API_URL`, `API_KEY`, `SECRET_KEY`, `MAX_UPLOAD_MB`) → same code in dev, Docker, cloud; no secrets in Git.
`python run.py` runs Flask with `debug=False`: the debug console allows arbitrary code execution from the
browser, tracebacks leak code/config, and the reloader would load the model twice. In Docker, gunicorn
(production WSGI server) runs the app. 📸 `/health` JSON.

## Chapter 5 — Database Trail
SQLAlchemy model `PredictionLog` (table `predictions`): id, `created_at` (UTC), sanitized filename,
image SHA-256, width/height, **label (what)**, **confidence + raw prob_ai + threshold (how confidently)**,
**model_name + model_version (which model)**, inference mode (local/remote), latency, client IP. One row
is committed per successful prediction; rejected uploads are not logged. PostgreSQL in Docker (named
volume `pgdata`), SQLite for local dev/tests — switched only via `DATABASE_URL`. History page shows newest
first, 20 per page, with per-label counts and mean confidence. 📸 history page; 📸 `psql` query in the db
container (docs/OPERATIONS.md).

## Chapter 6 — Docker
### 6.1 Warm-up
`labs/docker_warmup.sh` with nginx: pull, run with `-p 8080:80` and a bind mount, logs, exec shell,
named volume, ps/stop/rm/rmi. 📸 each. Definitions: **image** — a read-only template (filesystem + start
command); **container** — a running (or stopped) instance of an image with its own writable layer;
**layer** — one cached filesystem diff produced by a Dockerfile instruction, shared between images;
**volume** — storage managed outside the container's writable layer so data survives removal;
**port mapping** — forwarding a host port to a container port (`host:container`).

### 6.2 Dockerfile
`docker/Dockerfile`: multi-stage; stage 1 installs requirements into a venv; stage 2 copies only the venv
and code onto `python:3.12-slim`. Requirements are copied and installed before the source, so code edits
reuse the cached dependency layer. Non-root user `app`; `HEALTHCHECK` calls `/health`; `.dockerignore`
excludes `.env`, `.git`, data, mlruns; secrets only arrive as runtime env vars. **TensorFlow choice:**
`tensorflow-cpu` (no CUDA libraries) — inference is one image at a time and free cloud VMs have no GPU;
on Arm hosts `tensorflow` (CPU-only there) via `--build-arg TF_PACKAGE`. 📸 Dockerfile link,
`docker compose exec web whoami` → `app`.

### 6.3 Compose
See `docker-compose.yml` and docs/OPERATIONS.md. 📸 `docker compose ps` healthy; 📸 `nc -zv localhost
5432` refused. Model file mounted read-only (justification in OPERATIONS.md).

### 6.4 Image analysis
Table: naive [FILL] GB vs optimised [FILL] MB; largest layers [FILL from `docker history`]. Conclusion
[FILL].

### 6.5 Operations and troubleshooting
📸 start/stop/restart/logs/exec. Failure: [FILL wrong password or missing model], log excerpt, diagnosis,
fix (docs/OPERATIONS.md has the expected messages).

### 6.6 GitHub Actions
Toy: `labs/ci_toy/` in a throwaway repo, matrix Python 3.11/3.12; 📸 red run (broken assert), green run.
Project: `.github/workflows/ci.yml` (ruff → pytest → docker build). 📸 green run, badge in README.
Secrets: GitHub → Settings → Secrets and variables → Actions, referenced as `${{ secrets.NAME }}`;
never in the YAML because the file and its Git history are visible to everyone with repo access and
cannot be "un-committed"; GitHub masks secret values in logs.

## Chapter 7 — API Deployment
Pathway [A Oracle / C Cloud Run]. Research notes + step-by-step with 📸 (docs/DEPLOY.md).
**API URL:** [FILL]. Postman: 📸 health, predict (real), predict (AI), 401 without key, 415 bad file;
📸 server log lines for those requests. Tested from [FILL: mobile network / other machine].

## Chapter 8 — Integrating Flask and the API
Local `.env`: `INFERENCE_MODE=remote`, `API_URL`, `API_KEY`. `app/remote.py` posts the image bytes with
`X-API-Key` and maps the JSON back to the same `Prediction` object, so routes and DB code are identical
in both modes; API errors/timeouts → friendly 502 page. 📸 `/health` showing `model_api: ok`; 📸 ≥4 result
pages (real, AI, borderline, rejected file); 📸 History with `remote` rows.

## Lessons / what went wrong
[FILL: e.g. tensorboard missing for keras-tuner on TF 2.21; Oracle capacity errors; iptables blocking 8080;
CPU training time; …]

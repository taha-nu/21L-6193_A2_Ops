# Deploying the model API (Task 7) and connecting Flask to it (Task 8)

Architecture for Tasks 7–8 (as the assignment specifies: the model runs in the cloud, Flask runs on localhost):

```
 your laptop (localhost)                                    cloud VM / container service
 ┌──────────────────────────────┐   HTTPS/HTTP + X-API-Key   ┌────────────────────────────┐
 │ Flask UI  (INFERENCE_MODE=   │ ─────────────────────────► │ Model API (api/server.py)  │
 │           remote)            │   POST /api/v1/predict     │ gunicorn + TensorFlow-CPU  │
 │ Postgres/SQLite  ◄── history │ ◄───────────────────────── │ champion.keras baked in    │
 └──────────────────────────────┘   JSON verdict             └────────────────────────────┘
```

Everything below assumes you already ran `scripts/run_experiments.sh` (or at least
`python scripts/fetch_model.py`), so `models/champion.keras` and `models/serving_config.json` exist.

Pick **one** pathway. A is the default; C is the fallback if Oracle has no free capacity in your region.

---

## Pathway A — Oracle Cloud Always Free VM

### Research notes (check these in YOUR console; screenshot them for the report)
- **Free limits changed on 15 June 2026.** The Always Free Ampere A1 (Arm) allowance is now
  **2 OCPUs and 12 GB RAM** in total (1,500 OCPU-hours / 9,000 GB-hours per month), down from 4 OCPUs / 24 GB.
  Free-only accounts with bigger instances had them shut down until resized. That is still plenty for
  this API (TensorFlow-CPU + MobileNetV2 needs well under 2 GB).
- **Arm vs x86:** the Ampere A1 shape is **Arm64 (aarch64)**. The `tensorflow-cpu` wheel is x86-only,
  so build with `TF_PACKAGE=tensorflow==2.21.0` on Arm (it is CPU-only there anyway). The free AMD
  `VM.Standard.E2.1.Micro` is x86 but has only 1 GB RAM, which is too tight for TensorFlow.
- **"Out of host capacity"**: free A1 instances are often unavailable. Retry later or try another
  availability domain; otherwise use Pathway C.
- **Idle reclamation:** Oracle may reclaim Always Free instances that stay idle (very low CPU, network
  and memory use) for 7 days. Keep the demo instance in use around grading time, or upgrade the
  account to Pay-As-You-Go (still free within limits).
- **Two firewalls:** (1) the VCN **security list / NSG** in the cloud console and (2) the VM's own
  **iptables** rules (Oracle's Ubuntu images block everything except SSH by default). Port 8080 must be
  opened in **both**.

### Step by step
1. **Create the VM**: Console → Compute → Instances → *Create instance*. Name: `21L-6193_A2_Ops`.
   Image: *Canonical Ubuntu 24.04*. Shape: *Ampere → VM.Standard.A1.Flex*, 2 OCPU, 12 GB.
   Networking: create/assign a public IPv4. Upload or download the SSH key. *Create*.
   📸 instance details page with public IP and shape.
2. **Open port 8080 in the cloud firewall**: Instance → Subnet → Security List → *Add Ingress Rule*:
   source `0.0.0.0/0`, TCP, destination port `8080`. 📸
3. **SSH in** (`ssh -i key ubuntu@<public-ip>`) and **open port 8080 in iptables**:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8080 -j ACCEPT
   sudo netfilter-persistent save
   ```
4. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER && newgrp docker
   docker version
   ```
5. **Get the code and the model onto the VM**:
   ```bash
   git clone https://github.com/<you>/21L-6193_A2_Ops.git && cd 21L-6193_A2_Ops
   # copy the model from your laptop (models/ is git-ignored):
   #   scp -i key models/champion.keras models/serving_config.json ubuntu@<ip>:~/21L-6193_A2_Ops/models/
   # or fetch it from DagsHub on the VM: create .env with the MLFLOW_* variables, then
   #   pip install "mlflow>=2.15,<3" pyyaml python-dotenv && python3 scripts/fetch_model.py
   ```
6. **Configure and start**:
   ```bash
   cat > .env <<'ENV'
   API_KEY=<long random string, e.g. from: openssl rand -hex 24>
   TF_PACKAGE=tensorflow==2.21.0
   ENV
   docker compose -f docker-compose.api.yml up -d --build
   docker compose -f docker-compose.api.yml ps        # wait for "healthy"
   docker compose -f docker-compose.api.yml logs -f   # "Model API ready: VisionGuard v.."
   ```
7. **Test from another network** (phone hotspot, or https://reqbin.com): open
   `http://<public-ip>:8080/api/v1/health`. 📸 with the URL bar visible.
   **Your API URL for the report:** `http://<public-ip>:8080`

Optional HTTPS: point a free DuckDNS name at the IP and put Caddy in front (`caddy reverse-proxy --from <name>.duckdns.org --to :8080`), then open 443 instead of 8080 in both firewalls.

---

## Pathway C — Google Cloud Run (managed container)

Research notes: the Cloud Run free tier (per billing account, per month) is 2 million requests,
180,000 vCPU-seconds and 360,000 GiB-seconds, but only in `us-central1`, `us-east1` and `us-west1`;
a billing account is required. Instances scale to zero, so the first request after idle is slow
(cold start: TensorFlow import + model load ≈ 10–20 s). There are no volumes, which is why
`docker/Dockerfile.api` bakes the model into the image.

Compose → Cloud Run mapping: `api` service → Cloud Run service; `ports` → `--port 8080` +
the generated `*.run.app` HTTPS URL; `.env` → `--set-env-vars` / Secret Manager; `restart` and
health check → managed by Cloud Run; the `db` service would map to Cloud SQL, but our API is
stateless so none is needed.

```bash
gcloud auth login && gcloud config set project <project-id>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create visionguard --repository-format=docker --location=us-central1
# build in the cloud from the repo root (models/champion.keras must exist locally):
cp docker/Dockerfile.api Dockerfile
gcloud builds submit --tag us-central1-docker.pkg.dev/<project-id>/visionguard/api:v1 .
rm Dockerfile
gcloud run deploy vg-21l-6193-api \
  --image us-central1-docker.pkg.dev/<project-id>/visionguard/api:v1 \
  --region us-central1 --port 8080 --memory 2Gi --cpu 1 --min-instances 0 --max-instances 2 \
  --allow-unauthenticated --set-env-vars API_KEY=<secret>
```
(Cloud Run service names must be lowercase, so use e.g. `vg-21l-6193-api`.)
`--allow-unauthenticated` makes the URL reachable; the X-API-Key check in the app still protects `/predict`.
**API URL:** the `https://...run.app` printed at the end.

---

## Postman (Task 7 evidence)
1. Import `scripts/visionguard.postman_collection.json`.
2. Collection → Variables: `base_url` = your API URL, `api_key` = your key.
3. Run **Health**, then **Predict (with key)** with a real photo and with an AI image
   (Body → form-data → key `image`, type *File*), **Predict (no key)** (expect 401), and
   **Predict (text file)** (expect 415). 📸 each response with status code and time.
4. On the VM, `docker compose -f docker-compose.api.yml logs api` shows one line per request
   (`predict file=... label=... prob_ai=...`). 📸 that "logged result".

## Task 8 — Flask on localhost calling the cloud API
```bash
# in your local .env
INFERENCE_MODE=remote
API_URL=http://<public-ip>:8080        # or https://<service>.run.app
API_KEY=<same secret as on the server>
python run.py                           # http://127.0.0.1:5000
```
- `/health` should show `"model_api": "ok"` and the cloud model's name/version. 📸
- Upload several images (real photos, AI images, a .txt renamed to .jpg). 📸 each result page.
- Open **History**: every row has mode `remote`, the model version served by the cloud, time,
  decision and confidence. 📸
- Or run the whole local side in Docker: set the same three variables in `.env` and
  `docker compose up -d --build`; the Postgres history then lives in the `pgdata` volume.

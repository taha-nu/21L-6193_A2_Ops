# Docker operations runbook (Tasks 6.3 – 6.5)

## 6.3 Bring the stack up and prove it is wired correctly
```bash
cp .env.example .env            # set POSTGRES_PASSWORD, SECRET_KEY (long random strings)
python scripts/fetch_model.py   # models/champion.keras (mounted read-only into web)
docker compose up -d --build
docker compose ps               # 📸 both services "Up (healthy)"; db shows NO published port
```
**Service-name networking:** `docker compose exec web python -c "import socket; socket.create_connection(('db',5432),3); print('db reachable by name')"`

**Database port is closed from outside** (📸 both commands):
```bash
nc -zv localhost 5432           # Connection refused  (or: Test-NetConnection localhost -Port 5432 on Windows)
docker port visionguard-db-1    # prints nothing: no host mapping
```
**Readiness order:** `docker compose logs web | head` shows the web container started only after `db` turned healthy (Compose `depends_on: condition: service_healthy`).

**Model file: mounted, not built in (web).** Justification: the image stays the same when a new champion is promoted (fetch the new file, `docker compose restart web`), the model file never ends up in an image layer or registry, and CI can build the image without access to DagsHub. The cloud API image *does* bake the model in, because managed container platforms (Cloud Run) have no host folders to mount and a self-contained image is easier to move.

## 6.4 Image size analysis
```bash
docker build -f docker/Dockerfile.naive -t visionguard-web:naive .
docker build -f docker/Dockerfile       -t visionguard-web:slim  .
docker images visionguard-web                                   # 📸 the two sizes
docker history --no-trunc --format "{{.Size}}\t{{.CreatedBy}}" visionguard-web:slim | sort -rh | head -5
```
Expect the largest layer to be the virtualenv copy (TensorFlow ≈ 85% of it), then the base image.
Optimisations applied in `docker/Dockerfile`: slim base instead of full `python:3.12`, `tensorflow-cpu`
instead of `tensorflow`, `PIP_NO_CACHE_DIR=1`, multi-stage build (pip and build leftovers stay in the
builder stage), and `.dockerignore` (no data/, mlruns/, .git in the context). Report the before/after
numbers from `docker images` in a small table.

## 6.5 Day-to-day operations (📸 each)
```bash
docker compose stop            # stop both, keep containers
docker compose start
docker compose restart web
docker compose logs -f web     # follow logs  (Ctrl+C to leave)
docker compose logs --tail 50 db
docker compose exec web sh -c 'whoami; id; ls -l /app/models'   # shell in web -> user "app", not root
docker compose exec db psql -U visionguard -d visionguard -c "SELECT id, created_at, label, round(confidence::numeric,3), model_name, model_version FROM predictions ORDER BY id DESC LIMIT 5;"
docker compose down            # remove containers (volume pgdata survives)
```

### Break it on purpose #1 — wrong database password
Postgres sets the password only when the volume is first initialised. Change `POSTGRES_PASSWORD` in `.env` afterwards and the database still holds the old one:
```bash
sed -i 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=wrong-on-purpose/' .env
docker compose up -d --force-recreate web
docker compose ps              # web restarting / unhealthy
docker compose logs web | tail -20
```
Expected log evidence: `Database not ready (OperationalError), retry n/10` and finally
`RuntimeError: Could not connect to the database at DATABASE_URL`; `docker compose logs db` shows
`FATAL: password authentication failed for user "visionguard"`.
**Diagnosis:** the db is healthy (pg_isready does not check passwords) but rejects the web app's
credentials → the two sides disagree on the password. **Fix:** restore the original value in `.env`
and `docker compose up -d web` (or, if the old password is lost, `docker compose down -v` to
recreate the volume, which deletes history).

### Break it on purpose #2 — missing model file
```bash
mv models/champion.keras models/champion.keras.bak
docker compose restart web && docker compose logs web | tail -15
```
Expected: `FileNotFoundError: Model file not found at /app/models/champion.keras. Run scripts/fetch_model.py or mount the models/ folder.` and gunicorn `Worker failed to boot`.
**Fix:** `mv models/champion.keras.bak models/champion.keras && docker compose restart web`.

#!/usr/bin/env bash
# Task 1.4 (toy part) - DVC basics on a tiny CSV.
# Run it step by step (copy-paste each block) so you can screenshot every stage.
#   export DAGSHUB_USER=<your dagshub username>
#   export DAGSHUB_TOKEN=<your dagshub token>          # never commit this
#   export TOY_REPO=21L-6193_A2_Ops_dvc_toy             # create this empty repo on DagsHub first
set -euo pipefail

mkdir -p ~/dvc_toy && cd ~/dvc_toy

# 1. initialise Git and DVC
git init
dvc init                     # creates .dvc/ (config + cache) and .dvcignore
git commit -m "init dvc"

# 2. a small CSV, tracked by DVC (not by Git)
printf "id,value\n1,10\n2,20\n3,30\n" > data.csv
dvc add data.csv             # -> data.csv.dvc (pointer) + adds data.csv to .gitignore
cat data.csv.dvc             # md5 hash, size, path: THIS is what Git stores
git add data.csv.dvc .gitignore
git commit -m "data v1: 3 rows"

# 3. change it and commit version 2
printf "4,40\n5,50\n" >> data.csv
dvc add data.csv
git add data.csv.dvc
git commit -m "data v2: 5 rows"
git log --oneline

# 4. go back to version 1: Git restores the old pointer, DVC restores the matching data
git checkout HEAD~1 -- data.csv.dvc
dvc checkout
cat data.csv                 # 3 rows again
git checkout HEAD -- data.csv.dvc && dvc checkout   # back to v2

# 5. configure DagsHub storage as the DVC remote (S3-compatible endpoint)
git remote add origin "https://dagshub.com/${DAGSHUB_USER}/${TOY_REPO}.git"
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl "https://dagshub.com/${DAGSHUB_USER}/${TOY_REPO}.s3"
dvc remote modify origin --local access_key_id     "${DAGSHUB_TOKEN}"   # --local => .dvc/config.local,
dvc remote modify origin --local secret_access_key "${DAGSHUB_TOKEN}"   #   which is git-ignored
git add .dvc/config && git commit -m "configure dagshub dvc remote"
#   (alternative without DagsHub: dvc remote add -d origin /tmp/dvc-storage)

# 6. push data + code, delete local copy, pull it back
dvc push
git push -u origin HEAD:main     # username = DagsHub user, password = token
rm -rf data.csv .dvc/cache
ls                               # data.csv is gone
dvc pull
cat data.csv                     # restored from the remote

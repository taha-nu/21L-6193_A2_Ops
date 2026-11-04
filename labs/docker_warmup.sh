#!/usr/bin/env bash
# Task 6.1 - Docker warm-up with a ready-made image (nginx). Screenshot each step.
set -x

docker pull nginx:alpine                                   # download an IMAGE (stack of read-only layers)
docker image ls                                            # list images
docker history nginx:alpine                                # its LAYERS

mkdir -p ~/docker_warmup && echo "<h1>Hello from $(whoami) - VisionGuard warm-up</h1>" > ~/docker_warmup/index.html

# run a CONTAINER: -d background, -p host:container PORT MAPPING, -v VOLUME (bind mount)
docker run -d --name warmup -p 8080:80 -v ~/docker_warmup:/usr/share/nginx/html:ro nginx:alpine
docker ps                                                  # running containers
curl -s http://localhost:8080                              # served from the mounted folder

docker logs warmup                                         # read its logs (the curl request shows up)
docker exec -it warmup sh -c 'whoami; hostname; ls /usr/share/nginx/html; exit'   # shell inside

# named volume demo: data survives the container
docker volume create warmup_data
docker run --rm -v warmup_data:/data alpine sh -c 'echo persisted > /data/note.txt'
docker run --rm -v warmup_data:/data alpine cat /data/note.txt

docker stop warmup && docker ps -a                         # stopped but still listed
docker rm warmup && docker ps -a                           # removed
docker volume rm warmup_data
docker rmi nginx:alpine && docker image ls                 # remove the image

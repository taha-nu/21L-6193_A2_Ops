#!/usr/bin/env bash
# Task 2 + 3.3 + 3.5 - the full experiment grid. Each run differs from the reference run
# (mnv2-ref) in ONE meaningful change, so the comparison table isolates its effect.
# Needs .env with the DagsHub MLflow variables (otherwise logs to ./mlruns).
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=2

python -m src.train --model baseline --run-name baseline-cnn                         # Task 2.2
python -m src.train --model transfer --run-name mnv2-ref        --tracking manual    # Task 2.3 + 3.3 manual
python -m src.train --model transfer --run-name mnv2-autolog    --tracking autolog   # Task 3.3 autolog (same config)
python -m src.train --model transfer --run-name mnv2-lr3e-4     --lr 0.0003          # learning rate
python -m src.train --model transfer --run-name mnv2-noaug      --no-augment         # augmentation
python -m src.train --model transfer --run-name mnv2-frozen     --unfreeze 0         # frozen layers (no fine-tune)
python -m src.train --model transfer --run-name mnv2-unfreeze40 --unfreeze 40        # frozen layers (more unfrozen)
python -m src.train --model transfer --run-name mnv2-160px      --img-size 160       # image size
python -m src.train --model transfer --run-name effb0           --backbone efficientnet_b0   # backbone

python -m src.registry compare                    # table + reports/figures/run_comparison.png
python -m src.registry register-best --top 2      # 2 registry versions, best -> @champion
python scripts/fetch_model.py                     # models/champion.keras for serving

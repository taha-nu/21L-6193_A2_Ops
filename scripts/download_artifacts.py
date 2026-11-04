"""Download the report figures of every finished run from MLflow/DagsHub, for the PDF report.

    python scripts/download_artifacts.py          -> reports/runs/<run name>/*.png, *.json, *.md ...
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS, project_name, setup_mlflow  # noqa: E402


def main():
    mlflow = setup_mlflow()
    runs = mlflow.search_runs(experiment_names=[project_name()],
                              filter_string="attributes.status = 'FINISHED'")
    for _, r in runs.iterrows():
        name = r.get("tags.mlflow.runName") or r.run_id[:8]
        out = REPORTS / "runs" / name
        try:
            local = mlflow.artifacts.download_artifacts(run_id=r.run_id, artifact_path="reports")
        except Exception as e:  # noqa: BLE001 - e.g. the keras-tuner run has no reports/
            print(f"skip {name}: {type(e).__name__}")
            continue
        shutil.copytree(local, out, dirs_exist_ok=True)
        print(f"{name:20s} -> {out}")


if __name__ == "__main__":
    main()

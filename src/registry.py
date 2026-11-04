"""Task 3.5 / 3.6 - compare runs, register models, set the `champion` alias.

  python -m src.registry compare                         # table + chart of all runs
  python -m src.registry register --run-id <id>          # new version of VisionGuard
  python -m src.registry alias --version 2 --alias champion
  python -m src.registry register-best --metric test_roc_auc --top 2   # 2 versions, best = champion
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common import FIG_DIR, REPORTS, load_params, project_name, setup_mlflow  # noqa: E402

COLS = ["test_roc_auc", "test_f1", "test_precision", "test_recall", "test_fpr", "test_accuracy", "best_epoch"]


def runs_df(mlflow):
    df = mlflow.search_runs(experiment_names=[project_name()], filter_string="attributes.status = 'FINISHED'")
    keep = ["run_id", "tags.mlflow.runName", "params.model", "params.backbone", "params.lr",
            "params.img_size", "params.augment", "params.unfreeze", "params.tracking"]
    keep = [c for c in keep if c in df] + [f"metrics.{c}" for c in COLS if f"metrics.{c}" in df]
    out = df[keep].copy()
    out.columns = [c.split(".")[-1] for c in out.columns]
    return out.sort_values("test_roc_auc", ascending=False)


def compare(mlflow):
    df = runs_df(mlflow)
    REPORTS.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / "run_comparison.csv", index=False)
    print(df.round(4).to_markdown(index=False))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(df) + 2))
    labels = df["runName"].fillna(df["run_id"].str[:8])
    y = range(len(df))
    for i, (m, c) in enumerate([("test_roc_auc", "#2a7ab9"), ("test_f1", "#d9622b"), ("test_recall", "#6a9f3a")]):
        ax.barh([v + i * 0.27 for v in y], df[m], height=0.27, label=m, color=c)
    ax.set_yticks([v + 0.27 for v in y], labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Run comparison (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "run_comparison.png", dpi=120)
    print("saved", FIG_DIR / "run_comparison.png")


def register(mlflow, run_id: str, name: str):
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    print(f"registered {name} v{mv.version} from run {run_id}")
    return mv


def set_alias(mlflow, name: str, version: str, alias: str):
    mlflow.MlflowClient().set_registered_model_alias(name, alias, str(version))
    print(f"{name}@{alias} -> version {version}")


def main():
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compare")
    r = sub.add_parser("register")
    r.add_argument("--run-id", required=True)
    al = sub.add_parser("alias")
    al.add_argument("--version", required=True)
    al.add_argument("--alias", default="champion")
    b = sub.add_parser("register-best")
    b.add_argument("--metric", default="test_roc_auc")
    b.add_argument("--top", type=int, default=2)
    args = a.parse_args()

    mlflow = setup_mlflow()
    name = load_params()["registry"]["model_name"]
    if args.cmd == "compare":
        compare(mlflow)
    elif args.cmd == "register":
        register(mlflow, args.run_id, name)
    elif args.cmd == "alias":
        set_alias(mlflow, name, args.version, args.alias)
    else:
        df = runs_df(mlflow).sort_values(args.metric, ascending=False).head(args.top)
        versions = [register(mlflow, rid, name) for rid in reversed(df.run_id.tolist())]
        set_alias(mlflow, name, versions[-1].version, "champion")   # best one registered last
        if len(versions) > 1:
            set_alias(mlflow, name, versions[-2].version, "challenger")


if __name__ == "__main__":
    main()

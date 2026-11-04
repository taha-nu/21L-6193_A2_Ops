"""Task 2.7 (apply part) - Keras Tuner on VisionGuard.

Searches learning rate, dropout and number of unfrozen backbone layers with
RandomSearch (>= 8 trials). Each trial is a short single-stage fit, scored on
val_loss. Results -> reports/tuner_results.csv and one MLflow run "keras-tuner".

    python -m src.tune --trials 10 --epochs 6

Caveat for the report: picking the trial with the best validation score uses the
validation set for selection, so that score is optimistically biased (we tried many
settings and kept the luckiest). The untouched test set is the honest estimate.
"""
from __future__ import annotations

import argparse
import json

import keras
import keras_tuner as kt
import pandas as pd

from src.common import REPORTS, ROOT, load_params, set_seeds, setup_mlflow
from src.data import make_dataset
from src.models import build_transfer, compile_binary


def make_hypermodel(backbone: str, img_size: int):
    def build(hp: kt.HyperParameters):
        lr = hp.Float("lr", 1e-5, 3e-3, sampling="log")
        dropout = hp.Float("dropout", 0.0, 0.6, step=0.1)
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40, 60])
        model, base = build_transfer(backbone, img_size, dropout)
        if n_unfrozen:
            base.trainable = True
            for i, l in enumerate(base.layers):
                l.trainable = i >= len(base.layers) - n_unfrozen and not isinstance(
                    l, keras.layers.BatchNormalization)
        return compile_binary(model, lr)
    return build


def main():
    p = load_params()
    a = argparse.ArgumentParser()
    a.add_argument("--trials", type=int, default=10)
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--img-size", type=int, default=p["img_size"])
    args = a.parse_args()
    set_seeds(p["seed"])
    bb = p["transfer"]["backbone"]
    tr = make_dataset("train", args.img_size, p["batch_size"], p["seed"], p["augment"])
    va = make_dataset("val", args.img_size, p["batch_size"], p["seed"])

    tuner = kt.RandomSearch(make_hypermodel(bb, args.img_size), objective="val_loss",
                            max_trials=args.trials, seed=p["seed"], overwrite=True,
                            directory=str(ROOT / "tuner"), project_name="visionguard")
    tuner.search(tr, validation_data=va, epochs=args.epochs,
                 callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2)])

    rows = []
    for t in tuner.oracle.trials.values():
        if t.status != "COMPLETED":
            continue
        rows.append({"trial": t.trial_id, **t.hyperparameters.values,
                     "val_loss": t.metrics.get_best_value("val_loss"),
                     "val_auc": t.metrics.get_best_value("val_auc"),
                     "val_accuracy": t.metrics.get_best_value("val_accuracy")})
    df = pd.DataFrame(rows).sort_values("val_loss")
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "tuner_results.csv", index=False)
    print(df.round(5).to_markdown(index=False))

    # which hyper-parameter mattered most? rank correlation with val_loss
    imp = df[["lr", "dropout", "unfrozen_layers"]].corrwith(df.val_loss, method="spearman").abs()
    print("\n|Spearman correlation| with val_loss (higher = mattered more):\n", imp.round(3).to_string())
    best = tuner.get_best_hyperparameters(1)[0].values
    print("\nBest hyper-parameters:", best)

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="keras-tuner"):
        mlflow.set_tag("kind", "tuning")
        mlflow.log_params({f"best_{k}": v for k, v in best.items()} | {"trials": args.trials})
        mlflow.log_metric("best_val_loss", float(df.val_loss.min()))
        mlflow.log_artifact(str(REPORTS / "tuner_results.csv"))
        mlflow.log_text(json.dumps(imp.to_dict(), indent=2), "importance.json")


if __name__ == "__main__":
    main()

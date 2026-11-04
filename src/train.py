"""Tasks 2.2-2.5 and 3.3-3.5 - train one model and log everything to MLflow (DagsHub).

Examples
  python -m src.train --model baseline --run-name baseline-cnn
  python -m src.train --model transfer --run-name mnv2-manual   --tracking manual
  python -m src.train --model transfer --run-name mnv2-autolog  --tracking autolog
  python -m src.train --model transfer --lr 3e-4                --run-name mnv2-lr3e-4
  python -m src.train --model transfer --no-augment             --run-name mnv2-noaug
  python -m src.train --model transfer --unfreeze 40            --run-name mnv2-unfreeze40
  python -m src.train --model transfer --img-size 160           --run-name mnv2-160px
  python -m src.train --model transfer --backbone efficientnet_b0 --run-name effb0

Two tracking modes (Task 3.3):
  manual  - our MLflowEpochLogger callback logs exactly what we choose, with a global
            epoch counter that continues across the frozen and fine-tuning stages.
  autolog - mlflow.tensorflow.autolog() logs optimizer params, per-epoch metrics, the
            model summary and the model itself with zero code, but restarts the step
            counter on the second fit() call and knows nothing about our test metrics,
            threshold, tags or data version (we add those by hand in both modes).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src import evaluate as ev
from src.common import (MODELS_DIR, SPLIT_DIR, dvc_data_version, git_commit, load_params,
                        project_name, set_seeds, setup_mlflow)
from src.data import make_dataset
from src.models import build_baseline, build_transfer, compile_binary, layer_table, unfreeze


class MLflowEpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging. `offset` keeps the step count going across fit() calls."""

    def __init__(self, mlflow, offset: int = 0):
        super().__init__()
        self.mlflow, self.offset = mlflow, offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=self.offset + epoch + 1)


def parse_args(p: dict) -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    a.add_argument("--backbone", default=p["transfer"]["backbone"])
    a.add_argument("--lr", type=float)
    a.add_argument("--fine-tune-lr", type=float, default=p["transfer"]["fine_tune_lr"])
    a.add_argument("--epochs", type=int)
    a.add_argument("--fine-tune-epochs", type=int, default=p["transfer"]["fine_tune_epochs"])
    a.add_argument("--dropout", type=float)
    a.add_argument("--unfreeze", default=str(p["transfer"]["unfreeze"]),
                   help='"last_block", an integer number of layers, or 0 to skip fine-tuning')
    a.add_argument("--img-size", type=int, default=p["img_size"])
    a.add_argument("--batch-size", type=int, default=p["batch_size"])
    a.add_argument("--no-augment", action="store_true")
    a.add_argument("--seed", type=int, default=p["seed"])
    a.add_argument("--tracking", choices=["manual", "autolog"], default="manual")
    a.add_argument("--run-name")
    a.add_argument("--register", action="store_true", help="register the model in the MLflow registry")
    args = a.parse_args()
    sec = p["baseline"] if args.model == "baseline" else p["transfer"]
    args.lr = args.lr or sec["lr"]
    args.epochs = args.epochs or sec["epochs"]
    args.dropout = sec["dropout"] if args.dropout is None else args.dropout
    return args


def callbacks(p: dict, ckpt: Path, best_so_far: float | None = None) -> list:
    c = p["callbacks"]
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=c["early_stop_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=best_so_far, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=c["reduce_lr_factor"],
                                          patience=c["reduce_lr_patience"], min_lr=1e-7, verbose=1),
    ]


def merge(h1: dict, h2: dict) -> dict:
    return {k: list(h1.get(k, [])) + list(h2.get(k, [])) for k in h1}


def main() -> None:
    p = load_params()
    args = parse_args(p)
    set_seeds(args.seed)
    mlflow = setup_mlflow()
    aug = None if args.no_augment else p["augment"]

    train_ds = make_dataset("train", args.img_size, args.batch_size, args.seed, aug)
    val_ds = make_dataset("val", args.img_size, args.batch_size, args.seed)
    test_ds = make_dataset("test", args.img_size, args.batch_size, args.seed)

    run_name = args.run_name or f"{args.model}-{args.backbone if args.model == 'transfer' else 'cnn'}"
    if args.tracking == "autolog":
        # log_models=False: autolog would save the model at the end of EACH fit() call, not the
        # best checkpoint across both stages, so we log the final model ourselves below.
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False, silent=True)

    with mlflow.start_run(run_name=run_name) as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_dir = MODELS_DIR / "runs" / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = run_dir / "best.keras"

        mlflow.set_tags({"git_commit": git_commit(), "dvc_data_version": dvc_data_version(),
                         "seed": args.seed, "model_type": args.model, "tracking": args.tracking,
                         "project": project_name()})
        params = {k: v for k, v in vars(args).items() if k not in ("register", "run_name")}
        params.update(augment=json.dumps(aug), threshold_strategy=p["threshold"]["strategy"],
                      max_fpr=p["threshold"]["max_fpr"], train_size=len(pd.read_csv(SPLIT_DIR / "train.csv")))
        if args.model == "baseline":
            params = {k: v for k, v in params.items()
                      if k not in ("backbone", "fine_tune_lr", "fine_tune_epochs", "unfreeze")}
        mlflow.log_params(params)

        def cbs(offset, best=None):
            extra = [MLflowEpochLogger(mlflow, offset)] if args.tracking == "manual" else []
            return callbacks(p, ckpt, best) + extra

        stage1_epochs = None
        if args.model == "baseline":
            model = compile_binary(build_baseline(args.img_size, args.dropout), args.lr)
            (tmp / "layer_table.md").write_text(layer_table(model))
            hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs(0)).history
        else:
            model, base = build_transfer(args.backbone, args.img_size, args.dropout)
            compile_binary(model, args.lr)
            trainable_1 = [w.path for w in model.trainable_weights]
            (tmp / "trainable_stage1.txt").write_text("\n".join(trainable_1))
            (tmp / "layer_table.md").write_text(layer_table(model))
            print(f"Stage 1: backbone frozen, {len(trainable_1)} trainable weight tensors (head only)")
            hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs(0)).history
            stage1_epochs = len(hist["loss"])

            if args.unfreeze not in ("0", "none"):
                if ckpt.exists():
                    model = keras.models.load_model(ckpt)
                    base = model.get_layer(base.name)
                names = unfreeze(base, args.backbone, args.unfreeze)
                compile_binary(model, args.fine_tune_lr)  # recompile after changing trainable
                (tmp / "trainable_stage2.txt").write_text(
                    "# unfrozen backbone layers\n" + "\n".join(names) + "\n# + head: gap, dropout, prob_ai")
                mlflow.log_metric("stage2_unfrozen_layers", len(names))
                print(f"Stage 2: unfroze {len(names)} backbone layers: {names[:3]} ... {names[-3:]}")
                h2 = model.fit(train_ds, validation_data=val_ds, epochs=stage1_epochs + args.fine_tune_epochs,
                               initial_epoch=stage1_epochs,
                               callbacks=cbs(stage1_epochs, min(hist["val_loss"]))).history
                hist = merge(hist, h2)

        # best weights across both stages = the checkpoint with the lowest val_loss
        if ckpt.exists():
            model = keras.models.load_model(ckpt)
        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        mlflow.log_metrics({"best_epoch": best_epoch, "best_val_loss": float(min(hist["val_loss"])),
                            "epochs_run": len(hist["loss"])})
        print(f"Best epoch: {best_epoch} (val_loss={min(hist['val_loss']):.4f})")

        # threshold on VALIDATION, then evaluate on TEST exactly once
        yv, pv = ev.predict(model, val_ds)
        thr = ev.choose_threshold(yv, pv, p["threshold"]["strategy"], p["threshold"]["max_fpr"])
        val_m = ev.metrics(yv, pv, thr)
        yt, pt = ev.predict(model, test_ds)
        test_m = ev.metrics(yt, pt, thr)
        test_m05 = ev.metrics(yt, pt, 0.5)
        mlflow.log_metrics({f"val_{k}_at_thr": v for k, v in val_m.items() if k != "threshold"})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_metrics({f"test_{k}_at_0.5": test_m05[k] for k in ("accuracy", "precision", "recall", "f1")})
        print("TEST:", json.dumps({k: round(float(v), 4) for k, v in test_m.items()}))

        # ---- artifacts (Task 3.4) ----
        title = f"{run_name}"
        ev.plot_history(hist, tmp / "training_curves.png", best_epoch, title, stage1_epochs)
        if stage1_epochs and len(hist["loss"]) > stage1_epochs:
            s1 = {k: v[:stage1_epochs] for k, v in hist.items()}
            s2 = {k: v[stage1_epochs:] for k, v in hist.items()}
            ev.plot_history(s1, tmp / "curves_stage1_frozen.png",
                            int(np.argmin(s1["val_loss"])) + 1, f"{title} stage 1 (frozen)")
            ev.plot_history(s2, tmp / "curves_stage2_finetune.png",
                            int(np.argmin(s2["val_loss"])) + 1, f"{title} stage 2 (fine-tune)")
        ev.plot_confusion(yt, pt, thr, tmp / "confusion_matrix.png", f"{title} (test)")
        ev.plot_roc(yt, pt, thr, tmp / "roc_curve.png", f"{title} ROC (test)")
        test_df = pd.read_csv(SPLIT_DIR / "test.csv").assign(prob_ai=pt, pred=(pt >= thr).astype(int))
        test_df.to_csv(tmp / "test_predictions.csv", index=False)
        ev.save_json({**test_m, "best_epoch": best_epoch}, tmp / "test_metrics.json")
        serving = {"threshold": thr, "img_size": args.img_size, "model_type": args.model,
                   "backbone": args.backbone if args.model == "transfer" else None,
                   "classes": ["real", "ai"], "run_id": run.info.run_id}
        (tmp / "serving_config.json").write_text(json.dumps(serving, indent=2))
        (tmp / "requirements-freeze.txt").write_text(
            subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
        with open(tmp / "model_summary.txt", "w") as f:
            model.summary(print_fn=lambda s, **_: f.write(s + "\n"))
        for f in tmp.iterdir():
            mlflow.log_artifact(str(f), artifact_path="reports")
        for split in ("train", "val", "test"):
            mlflow.log_artifact(str(SPLIT_DIR / f"{split}.csv"), artifact_path="splits")

        # the model: plain .keras file (what the server loads) + MLflow model (for the registry)
        final = run_dir / "model.keras"
        model.save(final)
        (run_dir / "serving_config.json").write_text(json.dumps(serving, indent=2))
        mlflow.log_artifact(str(final), artifact_path="keras_model")
        mlflow.log_artifact(str(run_dir / "serving_config.json"), artifact_path="keras_model")
        from mlflow.models import infer_signature
        example = np.zeros((1, args.img_size, args.img_size, 3), "float32")
        info = mlflow.tensorflow.log_model(
            model, artifact_path="model",
            signature=infer_signature(example, model(example, training=False).numpy()))
        print(f"Run {run.info.run_id} done. Local copy: {final}")

        if args.register:
            mv = mlflow.register_model(info.model_uri, p["registry"]["model_name"])
            print(f"Registered {mv.name} version {mv.version}")


if __name__ == "__main__":
    main()

"""Task 2.4/2.5 - threshold choice, test metrics and figures.

Positive class = AI-generated. A FALSE POSITIVE is a real photo called fake.
For this product that is the worse error: a false accusation can damage a
photographer's or journalist's reputation, whereas a missed fake is a smaller harm
when the tool is only one signal among others. So the threshold is picked on the
VALIDATION set as the lowest value that keeps the false-positive rate at or below
params.threshold.max_fpr (default 5%), which gives the highest recall under that cap.
The test set is used once, with that fixed threshold.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def predict(model, ds) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in ds:
        ps.append(model(x, training=False).numpy().ravel())
        ys.append(y.numpy().ravel())
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def choose_threshold(y, p, strategy="max_fpr", max_fpr=0.05) -> float:
    if strategy == "f1":
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        return float(thr[np.argmax(f1[:-1])])
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    if strategy == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    ok = np.where(fpr <= max_fpr)[0]  # roc thresholds are decreasing -> last ok = lowest thr
    return float(thr[ok[-1]]) if len(ok) else 0.5


def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "fpr": fp / max(fp + tn, 1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": thr}


def plot_confusion(y, p, thr, out: Path, title="") -> None:
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "ai"], yticklabels=["real", "ai"],
           xlabel="predicted", ylabel="true", title=f"{title}\nthreshold={thr:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_roc(y, p, thr, out: Path, title="") -> None:
    fpr, tpr, t = roc_curve(y, p)
    k = np.argmin(np.abs(t - thr))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.scatter(fpr[k], tpr[k], color="red", zorder=3, label=f"chosen thr {thr:.2f}")
    ax.set(xlabel="False positive rate (real called AI)", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_history(history: dict, out: Path, best_epoch: int, title="", split_at: int | None = None):
    """Loss and accuracy curves, best epoch (lowest val_loss) marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(history["loss"]) + 1)
    for ax, m in zip(axes, ["loss", "accuracy"], strict=True):
        ax.plot(ep, history[m], label=f"train {m}")
        ax.plot(ep, history[f"val_{m}"], label=f"val {m}")
        ax.axvline(best_epoch, color="red", ls="--", label=f"best epoch {best_epoch}")
        if split_at:
            ax.axvline(split_at + 0.5, color="gray", ls=":", label="start fine-tuning")
        ax.set(xlabel="epoch", title=f"{title} {m}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_json(d: dict, out: Path) -> None:
    out.write_text(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                               for k, v in d.items()}, indent=2))

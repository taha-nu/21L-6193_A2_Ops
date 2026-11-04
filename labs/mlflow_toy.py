"""Tasks 3.1 + 3.2 - tiny Keras model on Fashion-MNIST, two learning rates, logged to MLflow.

Locally (3.1):      python labs/mlflow_toy.py            then:  mlflow ui   -> http://127.0.0.1:5000
On DagsHub (3.2):   put MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD in .env, run again.

Each call creates one run per learning rate inside the experiment "toy-fashion-mnist".
Select both runs in the UI and click "Compare".
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common import set_seeds, setup_mlflow  # noqa: E402

(x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
x_tr, x_te = x_tr[:10000] / 255.0, x_te[:2000] / 255.0   # small subset -> seconds per run
y_tr, y_te = y_tr[:10000], y_te[:2000]

mlflow = setup_mlflow("toy-fashion-mnist")
print("Tracking URI:", mlflow.get_tracking_uri())

for lr in [1e-3, 1e-2]:
    set_seeds(0)
    with mlflow.start_run(run_name=f"toy-lr-{lr}"):
        params = {"lr": lr, "units": 64, "epochs": 5, "batch_size": 128}
        mlflow.log_params(params)                          # PARAMETERS: inputs we chose
        mlflow.set_tags({"lab": "3.1", "dataset": "fashion_mnist",
                         "where": "dagshub" if "dagshub" in mlflow.get_tracking_uri() else "local"})
        model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                                  keras.layers.Dense(params["units"], activation="relu"),
                                  keras.layers.Dense(10, activation="softmax")])
        model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
        h = model.fit(x_tr, y_tr, validation_split=0.1, epochs=params["epochs"],
                      batch_size=params["batch_size"], verbose=2).history
        for e in range(params["epochs"]):                  # METRICS: numbers per step
            mlflow.log_metrics({k: h[k][e] for k in h}, step=e + 1)
        mlflow.log_metric("test_accuracy", model.evaluate(x_te, y_te, verbose=0)[1])
        fig, ax = plt.subplots()
        ax.plot(h["loss"], label="train")
        ax.plot(h["val_loss"], label="val")
        ax.set(title=f"loss, lr={lr}", xlabel="epoch")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")           # ARTIFACT: a file attached to the run
        mlflow.tensorflow.log_model(model, artifact_path="model")
print("Done. Open the UI and compare the two runs." + ("" if os.getenv("MLFLOW_TRACKING_URI") else
      "  Local UI:  mlflow ui"))

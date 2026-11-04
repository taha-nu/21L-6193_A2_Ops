"""Task 2.7 (toy part) - Keras Tuner Hyperband on Fashion-MNIST: learning rate + units.

    python labs/tuner_toy.py
"""
import keras
import keras_tuner as kt
import pandas as pd

(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, xt = x[:20000] / 255.0, xt / 255.0
y = y[:20000]


def build(hp):
    model = keras.Sequential([
        keras.Input((28, 28)), keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
                  "sparse_categorical_crossentropy", ["accuracy"])
    return model


tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=8, factor=3, seed=1,
                     directory="tuner_toy", project_name="fmnist", overwrite=True)
tuner.search(x, y, validation_split=0.2, verbose=1)

rows = [{"trial": t.trial_id, **t.hyperparameters.values,
         "val_accuracy": t.metrics.get_best_value("val_accuracy")}
        for t in tuner.oracle.trials.values() if t.status == "COMPLETED"]
df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
print(df.head(10).round(4).to_markdown(index=False))
best = tuner.get_best_hyperparameters(1)[0]
print("\nBest settings:", best.values)
m = tuner.get_best_models(1)[0]
print("Test accuracy of best model:", round(m.evaluate(xt, yt, verbose=0)[1], 4))

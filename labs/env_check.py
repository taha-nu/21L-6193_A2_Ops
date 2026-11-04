"""Task 1.1 - print exact library versions and whether TensorFlow sees a GPU.

    python labs/env_check.py     (screenshot this output with your terminal prompt visible)
"""
import getpass
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

print(f"user={getpass.getuser()}  host={platform.node()}  time={datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python      {sys.version.split()[0]}  ({sys.executable})")
print(f"platform    {platform.platform()}")
for pkg in ["tensorflow", "tensorflow-cpu", "keras", "keras-tuner", "mlflow", "dagshub", "dvc", "numpy",
            "pandas", "pillow", "scikit-learn", "matplotlib", "flask", "sqlalchemy"]:
    try:
        print(f"{pkg:12s}{version(pkg)}")
    except PackageNotFoundError:
        pass

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print("TF built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs visible to TensorFlow:", gpus if gpus else "none -> training will run on CPU")

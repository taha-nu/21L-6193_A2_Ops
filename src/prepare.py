"""Clean + split the dataset (feeds Task 2.1).

* drops unreadable files, exact duplicates and hashes that appear in both classes
* converts every remaining image to RGB JPEG (visionguard.preprocess.to_rgb_jpeg_bytes),
  so tf.data only ever has to decode JPEG and serving can do exactly the same
* stratified 70/15/15 split with a fixed seed -> data/splits/{train,val,test}.csv

Splitting AFTER de-duplication matters: a duplicate in both train and test would leak
the answer and inflate test accuracy.

    python -m src.prepare
"""
from __future__ import annotations

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.common import CLASSES, PROC_DIR, RAW_DIR, REPORTS, SPLIT_DIR, load_params
from visionguard.preprocess import to_rgb_jpeg_bytes

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    p = load_params()
    df = pd.read_csv(REPORTS / "audit.csv")
    keep = df[df.readable & ~df.duplicate & ~df.label_conflict & df["class"].isin(CLASSES)].copy()
    print(f"Keeping {len(keep)} of {len(df)} files")

    rows = []
    for r in keep.to_dict("records"):
        out = PROC_DIR / r["class"] / f"{r['sha256'][:16]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            with Image.open(RAW_DIR / r["path"]) as im:
                if getattr(im, "n_frames", 1) > 1:
                    im.seek(0)
                out.write_bytes(to_rgb_jpeg_bytes(im))
        rows.append({"filepath": out.relative_to(PROC_DIR).as_posix(), "label": CLASSES.index(r["class"]),
                     "class": r["class"], "source": r["path"], "width": r["width"], "height": r["height"]})
    data = pd.DataFrame(rows)

    s, seed = p["split"], p["seed"]
    train, rest = train_test_split(data, train_size=s["train"], stratify=data.label, random_state=seed)
    val, test = train_test_split(rest, test_size=s["test"] / (s["val"] + s["test"]),
                                 stratify=rest.label, random_state=seed)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("filepath").to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name:5s} {len(part):5d}  " + "  ".join(
            f"{c}={(part['class'] == c).sum()}" for c in CLASSES))


if __name__ == "__main__":
    main()

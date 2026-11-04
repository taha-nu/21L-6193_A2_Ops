"""Task 1.2 - Dataset audit.

Walks data/raw, and for every file records class, size, width, height, colour mode,
format and SHA-256. Detects unreadable files and exact (byte-identical) duplicates.

    python -m src.audit                    # -> reports/audit.csv, reports/audit_summary.md
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from src.common import RAW_DIR, REPORTS, class_from_folder

Image.MAX_IMAGE_PIXELS = None  # some real photos are huge; we only read headers/decode once


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"path": str(path), "class": class_from_folder(path), "file_size": path.stat().st_size,
           "sha256": sha256(path), "width": None, "height": None, "mode": None,
           "format": None, "readable": False, "error": ""}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.load()  # full decode: catches truncated / corrupt files that open() misses
        row["readable"] = True
    except Exception as e:  # noqa: BLE001 - we want every failure recorded
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def main(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = [p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
    df = pd.DataFrame(inspect(p) for p in files)
    df["path"] = df["path"].map(lambda s: Path(s).relative_to(raw_dir).as_posix())  # "/" on every OS

    # exact duplicates = same content hash. keep the first, flag the rest
    df["duplicate"] = df.duplicated("sha256", keep="first")
    # same bytes but different class folder = label conflict (dangerous for training)
    conflict = df.groupby("sha256")["class"].nunique()
    df["label_conflict"] = df["sha256"].map(conflict) > 1

    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "audit.csv", index=False)

    ok = df[df.readable]
    lines = ["# Dataset audit summary\n", f"Files scanned: {len(df)}\n",
             "## Images per class (all files / readable / unique readable)\n",
             "| class | files | readable | unique |", "|---|---|---|---|"]
    for c, g in df.groupby("class", dropna=False):
        lines.append(f"| {c} | {len(g)} | {g.readable.sum()} | {(g.readable & ~g.duplicate).sum()} |")
    lines += ["", "## Width / height / file size (readable images)\n",
              "| class | stat | width (px) | height (px) | file size (KB) |", "|---|---|---|---|---|"]
    for c, g in ok.groupby("class"):
        for name, fn in [("min", "min"), ("median", "median"), ("max", "max")]:
            lines.append(f"| {c} | {name} | {getattr(g.width, fn)():.0f} | {getattr(g.height, fn)():.0f} "
                         f"| {getattr(g.file_size, fn)() / 1024:.1f} |")
    lines += ["", f"Corrupt / unreadable files: **{(~df.readable).sum()}**",
              f"Exact duplicates (extra copies): **{df.duplicate.sum()}**",
              f"Hashes appearing in both classes: **{df[df.label_conflict].sha256.nunique()}**", "",
              "## Formats\n", ok.groupby(["class", "format"]).size().to_frame("n").to_markdown(),
              "", "## Colour modes\n", ok.groupby(["class", "mode"]).size().to_frame("n").to_markdown()]
    if (~df.readable).any():
        lines += ["", "## Unreadable files\n", df[~df.readable][["path", "error"]].to_markdown(index=False)]
    (REPORTS / "audit_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    main(ap.parse_args().raw_dir)

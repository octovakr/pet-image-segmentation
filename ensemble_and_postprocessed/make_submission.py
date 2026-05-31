"""Convert per-image .npy predictions into submission.csv for Kaggle.

Inputs
------
--pred_dir : directory containing one .npy per test image, named {id}.npy.
             Each file must be a uint8 2D array of shape (H, W) at the
             original image resolution, with values in {0, 1, 2}.
--sample   : path to sample_submission.csv (provides id list and H, W).
--out      : output CSV path.

Output
------
A CSV with columns: id, height, width, rle.
The rle column is multiclass run-length encoding in row-major order:
'v1 l1 v2 l2 ...' (value, run length).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes


def fill_boundary_enclosed_regions(mask: np.ndarray) -> np.ndarray:
    """boundary/foreground 로 둘러싸인 내부를 foreground(0)로 채우는 2단계 후처리.

    1단계 — boundary(2) 로 완전히 둘러싸인 내부의 비-boundary 픽셀을 0 으로 변경.
            boundary 클래스를 벽으로 보고, 이미지 경계와 연결되지 않은(즉 boundary
            로 에워싸인) 내부 영역을 채웁니다. binary_fill_holes 가 boundary 로
            둘러싸인 구멍을 메우므로, 메워진 픽셀 중 원래 boundary 가 아니었던
            곳이 곧 '둘러싸인 내부'입니다.
    2단계 — 1단계 결과에서 foreground(0) 가 만드는 loop 내부(즉 foreground 로
            완전히 둘러싸여 갇힌 background/boundary 픽셀)도 0 으로 변경합니다.
            이로써 동물 몸체 내부에 남은 잔여 boundary/background 점들이 흡수됩니다.
    """
    out = mask.copy()

    # 1단계: boundary 로 둘러싸인 비-boundary -> foreground
    boundary = (out == 2)
    filled_b = binary_fill_holes(boundary)
    out[filled_b & (~boundary)] = 0

    # 2단계: foreground 로 둘러싸인 내부(갇힌 bg/boundary) -> foreground
    fg = (out == 0)
    filled_f = binary_fill_holes(fg)
    out[filled_f & (~fg)] = 0

    return out


def multiclass_rle_encode(mask: np.ndarray) -> str:
    flat = mask.flatten().astype(np.int64)
    if flat.size == 0:
        return ""
    change = np.concatenate(([True], flat[1:] != flat[:-1]))
    starts = np.where(change)[0]
    values = flat[starts]
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    parts = []
    for v, l in zip(values.tolist(), lengths.tolist()):
        parts.append(str(int(v)))
        parts.append(str(int(l)))
    return " ".join(parts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--sample", type=str, default="sample_submission.csv")
    p.add_argument("--out", type=str, default="submission.csv")
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        print(f"ERROR: pred_dir not found: {pred_dir}", file=sys.stderr)
        sys.exit(1)

    sample_path = Path(args.sample)
    if not sample_path.is_file():
        print(f"ERROR: sample file not found: {sample_path}", file=sys.stderr)
        sys.exit(1)

    expected = []
    with open(sample_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "height", "width", "rle"}
        if required - set(reader.fieldnames or []):
            print(f"ERROR: sample CSV missing columns. Got {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            expected.append((row["id"].strip(), int(row["height"]), int(row["width"])))

    out_rows = []
    missing, bad_shape, bad_values = [], [], []
    for sid, H, W in expected:
        npy_path = pred_dir / f"{sid}.npy"
        if not npy_path.exists():
            missing.append(sid)
            continue
        pred = np.load(npy_path)
        if pred.shape != (H, W):
            bad_shape.append((sid, pred.shape, (H, W)))
            continue
        unique = np.unique(pred)
        if not np.all((unique >= 0) & (unique <= 2)):
            bad_values.append((sid, unique.tolist()))
            continue
        # 후처리: boundary(2)로 완전히 둘러싸인 영역의 내부를 foreground(0)로 변경
        pred = fill_boundary_enclosed_regions(pred.astype(np.uint8))
        out_rows.append((sid, H, W, multiclass_rle_encode(pred)))

    if missing:
        print(f"ERROR: missing prediction .npy for {len(missing)} ids "
              f"(e.g. {missing[:3]}).", file=sys.stderr)
        sys.exit(2)
    if bad_shape:
        print(f"ERROR: shape mismatch on {len(bad_shape)} predictions. "
              f"Predictions must be at original resolution. "
              f"First few: {bad_shape[:3]}", file=sys.stderr)
        sys.exit(2)
    if bad_values:
        print(f"ERROR: out-of-range labels in {len(bad_values)} predictions. "
              f"Values must be in {{0, 1, 2}}. First few: {bad_values[:3]}",
              file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "height", "width", "rle"])
        w.writerows(out_rows)
    print(f"Wrote {out_path}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

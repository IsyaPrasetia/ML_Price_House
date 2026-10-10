"""Multiple Linear Regression murni Python (gradient descent) — tanpa numpy/sklearn.

Arsitektur ML mini:
  data/sample.csv -> ml/train.py -> model/model.json -> ml/predict.py
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["luas_tanah_m2", "luas_bangunan_m2", "kamar_mandi", "jarak_stasiun_km"]
TARGET = "harga_juta_idr"


def load_dataset(path: Path) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append([float(row[c]) for c in FEATURES])
            ys.append(float(row[TARGET]))
    return xs, ys


def zscore_fit(xs: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(xs)
    means = [sum(r[i] for r in xs) / n for i in range(len(FEATURES))]
    stds = [
        math.sqrt(sum((r[i] - means[i]) ** 2 for r in xs) / n) or 1.0
        for i in range(len(FEATURES))
    ]
    return means, stds


def zscore(xs: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    return [[(r[i] - means[i]) / stds[i] for i in range(len(FEATURES))] for r in xs]


def train(xs_n: list[list[float]], ys: list[float], epochs: int = 4000, lr: float = 0.01):
    random.seed(42)
    w = [0.0] * len(FEATURES)
    b = 0.0
    n = len(xs_n)
    for _ in range(epochs):
        pred = [sum(w[i] * r[i] for i in range(len(w))) + b for r in xs_n]
        err = [p - y for p, y in zip(pred, ys)]
        grad_w = [sum(err[k] * xs_n[k][i] for k in range(n)) / n for i in range(len(w))]
        grad_b = sum(err) / n
        w = [wi - lr * g for wi, g in zip(w, grad_w)]
        b -= lr * grad_b
    mse = sum(e**2 for e in err) / n
    return w, b, mse


def r2_score(ys: list[float], preds: list[float]) -> float:
    mean_y = sum(ys) / len(ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - mean_y) ** 2 for y in ys) or 1.0
    return 1 - ss_res / ss_tot


def main() -> None:
    xs, ys = load_dataset(ROOT / "data" / "sample.csv")
    means, stds = zscore_fit(xs)
    xs_n = zscore(xs, means, stds)

    # hold-out 80/20
    split = int(len(xs_n) * 0.8)
    w, b, mse = train(xs_n[:split], ys[:split])
    preds = [sum(w[i] * r[i] for i in range(len(w))) + b for r in xs_n[split:]]
    r2 = r2_score(ys[split:], preds)

    model = {
        "features": FEATURES,
        "weights": w,
        "bias": b,
        "norm": {"means": means, "stds": stds},
        "metrics": {"mse_train": round(mse, 2), "r2_holdout": round(r2, 4)},
    }
    out = ROOT / "model" / "model.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Model tersimpan -> {out}")
    print(f"MSE (train)     : {mse:,.2f} juta^2")
    print(f"R^2 (holdout)   : {r2:.4f}")


if __name__ == "__main__":
    main()

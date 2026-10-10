"""Prediksi harga rumah dari model/model.json (CLI)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict(features: list[float], model: dict) -> float:
    means = model["norm"]["means"]
    stds = model["norm"]["stds"]
    z = [(f - m) / s for f, m, s in zip(features, means, stds)]
    return sum(w * x for w, x in zip(model["weights"], z)) + model["bias"]


def main() -> None:
    p = argparse.ArgumentParser(description="Prediksi harga rumah (juta IDR)")
    for f in ["luas_tanah_m2", "luas_bangunan_m2", "kamar_mandi", "jarak_stasiun_km"]:
        p.add_argument(f, type=float)
    args = p.parse_args()

    model = json.loads((ROOT / "model" / "model.json").read_text(encoding="utf-8"))
    x = [args.luas_tanah_m2, args.luas_bangunan_m2, args.kamar_mandi, args.jarak_stasiun_km]
    harga = predict(x, model)
    m = model["metrics"]
    print(f"Estimasi harga : Rp {harga:,.0f} juta")
    print(f"(model: R2 holdout {m['r2_holdout']})")


if __name__ == "__main__":
    main()

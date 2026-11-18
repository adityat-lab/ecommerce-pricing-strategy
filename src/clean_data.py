"""Step 1: clean the retail price data and engineer pricing features."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/clean.csv")
CANDIDATES = ["retail_price.csv", "sample_retail_price.csv"]

RENAME = {
    "productid": "product_id", "productcategoryname": "category", "monthyear": "date",
    "qty": "qty", "totalprice": "revenue", "freightprice": "freight_price",
    "unitprice": "unit_price", "productscore": "product_score", "customers": "customers",
    "weekend": "weekend", "holiday": "holiday", "comp1": "comp_1", "comp2": "comp_2",
    "comp3": "comp_3", "lagprice": "lag_price",
}
OPTIONAL = ["freight_price", "product_score", "customers", "weekend", "holiday", "lag_price"]


def norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def find_raw_file() -> Path:
    for name in CANDIDATES:
        if (RAW_DIR / name).exists():
            return RAW_DIR / name
    raise SystemExit("No raw file in data/raw/. Download 'retail_price.csv' from Kaggle "
                     "(see README) or run: python src/generate_sample_data.py")


def main():
    path = find_raw_file()
    print(f"Reading {path}")
    df = pd.read_csv(path)
    df.columns = [RENAME.get(norm(c), norm(c)) for c in df.columns]
    n0 = len(df)

    need = ["product_id", "category", "date", "qty", "unit_price"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.drop_duplicates()
    for c in ["product_id", "category"]:
        df[c] = df[c].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for c in df.columns:
        if c not in ("product_id", "category", "date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "qty", "unit_price"])
    df = df[(df.qty > 0) & (df.unit_price > 0)]
    if "revenue" not in df.columns:
        df["revenue"] = np.nan
    df["revenue"] = df["revenue"].fillna(df.qty * df.unit_price)

    for c in ["comp_1", "comp_2", "comp_3"]:
        if c not in df.columns:
            df[c] = np.nan
    df["comp_avg"] = df[["comp_1", "comp_2", "comp_3"]].mean(axis=1)
    df["price_gap_pct"] = ((df.unit_price - df.comp_avg) / df.comp_avg * 100).round(2)

    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(df[c].median()).fillna(0)

    # drop extreme price outliers inside each product (more than 3x or less than 1/3 of its median)
    med = df.groupby("product_id")["unit_price"].transform("median")
    df = df[(df.unit_price <= 3 * med) & (df.unit_price >= med / 3)]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    keep = ["product_id", "category", "date", "year", "month", "qty", "unit_price", "revenue",
            "freight_price", "product_score", "customers", "weekend", "holiday", "lag_price",
            "comp_1", "comp_2", "comp_3", "comp_avg", "price_gap_pct"]
    df = df[keep].sort_values(["product_id", "date"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Rows: {n0} raw -> {len(df)} clean | products: {df.product_id.nunique()} | "
          f"categories: {df.category.nunique()} | {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

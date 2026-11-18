"""Step 6: load clean data into SQLite and run sql/analysis_queries.sql"""
import sqlite3
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/clean.csv")
with sqlite3.connect("data/pricing.db") as con:
    df.to_sql("sales", con, if_exists="replace", index=False)
    text = Path("sql/analysis_queries.sql").read_text(encoding="utf-8")
    for block in [b.strip() for b in text.split(";") if b.strip()]:
        print(f"\n=== {block.splitlines()[0].lstrip('- ').strip()} ===")
        print(pd.read_sql_query(block, con).to_string(index=False))

#!/usr/bin/env python3
"""
evaluate.py – 5‑fold cross‑validation metrics for GHRS
------------------------------------------------------

• Loads the per‑fold **predicted_ratings.pkl** files produced by
  `rank_model.py --fold F`  (F = 1…5).

• Computes Root‑Mean‑Squared‑Error, Precision and Recall exactly as
  described in §4 .3 of the paper.

Outputs
-------
<root>/evaluation_metrics.csv   (human‑readable summary)
<root>/evaluation_metrics.pkl   (pickled python dict, optional)

usage
=====
python evaluate.py                                  # default paths
python evaluate.py --root data100k --out my.csv     # custom names
"""
from __future__ import annotations
import argparse, pickle, numpy as np, pandas as pd
from math import sqrt
from pathlib import Path
from sklearn.metrics import mean_squared_error, precision_score, recall_score

# ----------------------------- CLI -------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--root", default="data100k",
               help="directory that contains f1/ … f5/ sub‑folders")
P.add_argument("--ml",   default="datasets/ml-100k",
               help="MovieLens‑100K directory with u*.test files")
P.add_argument("--out",  default="evaluation_metrics.csv",
               help="name of the CSV file to create inside --root")
P.add_argument("--pickle", action="store_true",
               help="also dump a pickled .pkl copy next to the CSV")
args = P.parse_args()

ROOT = Path(args.root)
ML   = Path(args.ml)

# ----------------------------- evaluation ------------------------------
rows = []           # collect per‑fold stats

for F in range(1, 6):
    pred_path = ROOT / f"f{F}" / "predicted_ratings.pkl"
    if not pred_path.exists():
        print(f"[warn] missing {pred_path}  →  skipping fold {F}")
        continue

    predU = pickle.load(open(pred_path, "rb"))

    test = pd.read_csv(ML / f"u{F}.test", sep="\t",
                       names=["uid", "iid", "r", "ts"]).drop(columns="ts")
    test["uid"] -= 1
    test["iid"] -= 1

    y_true = test["r"].to_numpy()
    y_pred = predU[test["uid"], test["iid"]]

    rmse  = sqrt(mean_squared_error(y_true, y_pred))
    rel_t = (y_true >= 4).astype(int)
    rel_p = (y_pred >= 4).astype(int)

    precision = precision_score(rel_t, rel_p, zero_division=0)
    recall    = recall_score   (rel_t, rel_p, zero_division=0)

    rows.append([F, rmse, precision, recall])

    print(f"Fold {F}: RMSE={rmse:.4f}  Precision={precision:.3f}  "
          f"Recall={recall:.3f}")

# ----------------------------- summary ---------------------------------
df = pd.DataFrame(rows, columns=["fold", "rmse", "precision", "recall"])
avg = df.mean(numeric_only=True)
df = pd.concat([df,
                pd.DataFrame([["AVG", avg.rmse, avg.precision, avg.recall]],
                             columns=df.columns)],
               ignore_index=True)

csv_path = ROOT / args.out
df.to_csv(csv_path, index=False, float_format="%.5f")
print("\n==== 5‑Fold Averages ====")
print(f"RMSE : {avg.rmse:.4f} ± {df['rmse'].std():.4f}")
print(f"Precision : {avg.precision:.3f}")
print(f"Recall    : {avg.recall:.3f}")
print(f"✓ metrics saved → {csv_path}")

if args.pickle:
    pkl_path = csv_path.with_suffix(".pkl")
    pickle.dump(df.to_dict(orient="list"), open(pkl_path, "wb"))
    print(f"✓ pickle  saved → {pkl_path}")

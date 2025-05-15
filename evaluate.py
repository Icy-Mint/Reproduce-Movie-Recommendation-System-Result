#!/usr/bin/env python3
"""
evaluate.py – 5‑fold cross‑validation metrics for GHRS
------------------------------------------------------

• Loads the per‑fold **predicted_ratings.pkl** files produced by
  `rank_model.py --fold F` (F = 1…5).

• Computes Root‑Mean‑Squared‑Error, Precision and Recall exactly as
  described in §4 .3 of the paper:
    – An item is **relevant** if the true rating ≥ 4.
    – Precision / Recall use the same 4‑star threshold on predictions.

usage
=====
    python evaluate.py                    # default paths
    python evaluate.py --root data100k    # if you changed folder names
"""

# ---------------------------------------------------------------- imports
import argparse, pickle, numpy as np, pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, precision_score, recall_score

# ---------------------------------------------------------------- CLI
P = argparse.ArgumentParser()
P.add_argument("--root", default="data100k",          # folder with f1/ … f5/
               help="root directory that contains f1/ … f5/ sub‑folders")
P.add_argument("--ml",   default="datasets/ml-100k",  # MovieLens files
               help="MovieLens‑100K dataset directory")
args = P.parse_args()

# ---------------------------------------------------------------- eval
rmse, prec, rec = [], [], []
for F in range(1, 6):
    predU = pickle.load(open(f"{args.root}/f{F}/predicted_ratings.pkl", "rb"))

    test  = pd.read_csv(f"{args.ml}/u{F}.test", sep="\t",
                        names=["uid", "iid", "r", "ts"]).drop(columns="ts")
    test["uid"] -= 1
    test["iid"] -= 1

    y_true = test.r.to_numpy()
    y_pred = predU[test.uid, test.iid]

    rmse.append(sqrt(mean_squared_error(y_true, y_pred)))

    rel_true = (y_true >= 4).astype(int)          # relevant ↔ rating ≥ 4
    rel_pred = (y_pred >= 4).astype(int)

    prec.append(precision_score(rel_true, rel_pred, zero_division=0))
    rec .append(recall_score   (rel_true, rel_pred, zero_division=0))

    print(f"Fold {F}: RMSE={rmse[-1]:.4f}  "
          f"Precision={prec[-1]:.3f}  Recall={rec[-1]:.3f}")

# ---------------------------------------------------------------- report
print("\n==== 5‑Fold Averages ====")
print(f"RMSE : {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
print(f"Precision : {np.mean(prec):.3f}")
print(f"Recall    : {np.mean(rec):.3f}")

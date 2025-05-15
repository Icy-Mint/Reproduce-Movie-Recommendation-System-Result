#!/usr/bin/env python
"""
cold_start.py — evaluate GHRS under synthetic cold‑start
(percentage‑wise removal of ratings from the *test* splits).

Examples
--------
Earliest‑timestamp deletion, 0…80 % in 10 % steps
$ python cold_start.py

Random deletion, 0…85 % in 5 % steps
$ python cold_start.py --policy random --step 5
"""
from __future__ import annotations
import argparse, pickle
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# ---------- CLI --------------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--root",   default="data100k",
               help="folder with f*/predicted_ratings.pkl")
P.add_argument("--ml",     default="datasets/ml-100k",
               help="MovieLens‑100K directory")
P.add_argument("--step",   type=int, default=10,
               help="percentage step size (default 10 → 0,10,20 … 80)")
P.add_argument("--policy", choices=["earliest", "random"], default="earliest",
               help="which ratings to delete first inside each test split")
P.add_argument("--out_csv", default="Figs/cold_start_rmse.csv")
args = P.parse_args()

ROOT  = Path(args.root)
ML    = Path(args.ml)
FOLDS = range(1, 6)
PCTS  = list(range(0, 90, args.step))                  # 0 … 80 %

# ---------- evaluation -------------------------------------------------
rmse_mat: list[list[float]] = []

for pct in tqdm(PCTS, desc="cold‑start %"):
    fold_scores = []
    for f in FOLDS:
        pred_file = ROOT / f"f{f}" / "predicted_ratings.pkl"
        if not pred_file.exists():
            continue

        predU = pickle.load(open(pred_file, "rb"))      # (U × I)

        test = pd.read_csv(ML / f"u{f}.test", sep="\t",
                           names=["uid", "iid", "r", "ts"])

        # -------- simulate cold‑start ----------------------------------
        n_drop = int(len(test) * pct / 100)
        if n_drop:
            if args.policy == "earliest":
                test = test.sort_values("ts", ascending=True).iloc[n_drop:]
            else:                                       # random removal
                test = test.sample(frac=1 - pct/100, random_state=42)

        if test.empty:
            continue

        # 0‑based ids expected by prediction matrix
        test["uid"] -= 1
        test["iid"] -= 1

        # keep only pairs that lie inside predU’s bounds
        keep = (test["uid"] < predU.shape[0]) & (test["iid"] < predU.shape[1])
        if keep.sum() == 0:
            continue

        y_true = test.loc[keep, "r"].to_numpy()
        coords = test.loc[keep, ["uid", "iid"]].to_numpy(dtype=int)
        y_pred = predU[coords[:, 0], coords[:, 1]]

        fold_scores.append(sqrt(mean_squared_error(y_true, y_pred)))
    rmse_mat.append(fold_scores)

# ---------- report -----------------------------------------------------
print("\n===== Cold‑Start RMSE (delete‑% → mean ± std over folds) =====")
rows = []
for pct, scores in zip(PCTS, rmse_mat):
    if scores:
        mu, sd, n = np.mean(scores), np.std(scores), len(scores)
        print(f"{pct:2d}% removed → RMSE {mu:.4f} ± {sd:.4f}  (n={n})")
        rows.append([pct, mu, sd, n])
    else:
        print(f"{pct:2d}% removed → —  (no valid folds)")
        rows.append([pct, np.nan, np.nan, 0])

# ---------- CSV --------------------------------------------------------
out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["pct_removed","rmse_mean","rmse_std","folds"])\
  .to_csv(out_csv, index=False)
print(f"\n✓ CSV saved → {out_csv}")

# ---------- plot -------------------------------------------------------
means = [r[1] for r in rows]
errs  = [r[2] for r in rows]

plt.figure(figsize=(6,4))
plt.errorbar(PCTS, means, yerr=errs, fmt="-o", capsize=4)
plt.title(f"Cold‑Start ({args.policy} deletion) • MovieLens 100K")
plt.xlabel("% ratings removed from test split")
plt.ylabel("RMSE")
plt.grid(ls="--", alpha=.4)
plt.tight_layout()

fig_path = Path("Figs/cold_start_rmse.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
plt.show(block=False)       # comment out if running headless
print(f"✓ figure saved → {fig_path}")

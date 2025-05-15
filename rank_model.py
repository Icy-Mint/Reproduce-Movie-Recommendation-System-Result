#!/usr/bin/env python
"""
rank_model.py  –  GHRS preference‑based ranking
Implements Algorithm 1 (steps 10‑19) on the FULL MovieLens‑100K set.

Requires
--------
encoded_features.pkl   (from autoencoder.py)
user_clusters.pkl      (from cluster_user.py – “whole‑set” mode)

Produces
--------
data100k/predicted_ratings.pkl   (shape 943 × 1682, float32)

Typical call
------------
python rank_model.py --alpha 0.01 --th 0.8
"""
# ----------------------------------------------------------------------
import os, random, argparse, pickle, numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# reproducibility (same seeds the paper used)
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42);  np.random.seed(42);  tf.random.set_seed(42)

# ------------------------ CLI -----------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--alpha", type=float, default=0.01,
                 help="alpha threshold (paper optimum 0.01)")
cli.add_argument("--th",    type=float, default=0.80,
                 help="item‑similarity threshold (paper used 0.80)")
cli.add_argument("--root",  default="data100k",
                 help="folder with *_features.pkl & *_clusters.pkl")
cli.add_argument("--ml",    default="datasets/ml-100k",
                 help="MovieLens‑100K directory (contains u.data)")
args = cli.parse_args()

ROOT, ALPHA, TH = args.root, args.alpha, args.th

# ------------------------ load material -------------------------------
clusters = np.asarray(pickle.load(open(f"{ROOT}/user_clusters.pkl", "rb")))
K = clusters.max() + 1                           # number of clusters

ratings = pd.read_csv(f"{args.ml}/u.data", sep='\t',
                      names=["uid", "iid", "r", "ts"]).drop(columns="ts")
ratings["uid"] -= 1;  ratings["iid"] -= 1
ratings["c"]    = clusters[ratings.uid]          # assign cluster to each row

U = ratings.uid.max() + 1                        # 943 users
I = ratings.iid.max() + 1                        # 1682 items
print(f"users={U}  items={I}  clusters={K}")

# ---------------- C × I mean matrix  (Alg. 1 step 11) -----------------
CI = (ratings
      .groupby(["c", "iid"])["r"].mean()
      .unstack(fill_value=np.nan)
      .reindex(index=range(K), columns=range(I)))
Cmean = np.nanmean(CI.to_numpy(), 1, keepdims=True)   # step 17 fallback

# ---------------- cluster‑local item similarities ---------------------
print("▸ computing cluster‑local item similarities")
simM = []
global_mean = np.nanmean(ratings.r)                   # ≈ 3.53  (no NaN)

for c in tqdm(range(K)):
    sub  = ratings[ratings.c == c]
    mat  = np.full((I, U), np.nan, dtype=np.float32)  # item × user
    mat[sub.iid, sub.uid] = sub.r

    means = np.nanmean(mat, 1, keepdims=True)         # I × 1
    # rows with *all* NaN get means==NaN – fix by global mean -------------
    means[np.isnan(means)] = global_mean
    filled = np.where(np.isnan(mat), means, mat)      # finite matrix

    simM.append(cosine_similarity(filled))            # I × I per cluster

# ---------------- fill missing cells (Alg. 1 13‑17) -------------------
predC = CI.to_numpy()                                 # will be modified
print("▸ filling missing cluster‑item cells")
for c in tqdm(range(K)):
    row   = predC[c]
    sim   = simM[c]
    holes = np.isnan(row)
    if not holes.any():                               # row already full
        continue
    for i in np.where(holes)[0]:
        neigh_mask = sim[i] >= TH;  neigh_mask[i] = False
        vals   = row[neigh_mask]
        wts    = sim[i, neigh_mask]
        good   = ~np.isnan(vals)
        predC[c, i] = (np.average(vals[good], weights=wts[good])
                       if good.any() else Cmean[c, 0])

# ---------------- user‑level matrix (Alg. 1 step 19) ------------------
predU = predC[clusters].astype(np.float32)             # 943 × 1682
out_file = f"{ROOT}/predicted_ratings.pkl"
pickle.dump(predU, open(out_file, "wb"))
print("✓ predicted_ratings.pkl saved →", out_file)

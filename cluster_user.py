#!/usr/bin/env python
"""
cluster_user.py – builds the user→cluster map used in GHRS
Includes Elbow and Silhouette analysis + PCA visualization of clusters.
"""

import argparse, pickle, numpy as np, pathlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------- CLI ----------------
P = argparse.ArgumentParser()
P.add_argument("--enc", default="data100k/encoded_features.pkl")
P.add_argument("--k",   type=int, default=8, help="number of clusters (paper: 8)")
P.add_argument("--out", default="data100k/user_clusters.pkl")
P.add_argument("--fig_dir", default="Figs")
args = P.parse_args()

# ---------------- Load Features ----------------
X = pickle.load(open(args.enc, "rb"))  # shape (943, 4)
pathlib.Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

# ---------------- Elbow Method ----------------
inertias = []
k_range = range(2, 21)
for k in k_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.savefig(f"{args.fig_dir}/elbow_curve.png")
plt.close()
print(f"✓ elbow_curve.png saved → {args.fig_dir}/elbow_curve.png")

# ---------------- Silhouette Scores ----------------
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

plt.figure()
plt.plot(k_range, sil_scores, marker="o", color="darkorange")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.savefig(f"{args.fig_dir}/silhouette_scores.png")
plt.close()
print(f"✓ silhouette_scores.png saved → {args.fig_dir}/silhouette_scores.png")

# ---------------- Final KMeans Clustering ----------------
kmeans = KMeans(n_clusters=args.k, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X).astype(np.int16)
pickle.dump(labels, open(args.out, "wb"))
print(f"✓ user_clusters.pkl saved — k={args.k} → {args.out}")

# ---------------- PCA Cluster Visualization ----------------
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

plt.figure(figsize=(7, 5))
for c in range(args.k):
    plt.scatter(*X_2D[labels == c].T, label=f"Cluster {c}", s=20)
plt.title(f"PCA of Encoded Features (k={args.k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{args.fig_dir}/clusters_pca_k{args.k}.png")
plt.close()
print(f"✓ cluster PCA plot saved → {args.fig_dir}/clusters_pca_k{args.k}.png")

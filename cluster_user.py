import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load encoded user features
with open('data100k/encoded_features.pkl', 'rb') as f:
    X_encoded = pickle.load(f)

# Ensure NumPy array of float32
X = np.array(X_encoded, dtype=np.float32)

# K values to test
K_RANGE = range(2, 15)
inertias = []
silhouettes = []

print(" Evaluating KMeans clustering for k =", list(K_RANGE))

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    try:
        sil_score = silhouette_score(X, labels)
    except ValueError:
        sil_score = float('nan')
    silhouettes.append(sil_score)

# Plot elbow and silhouette
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_RANGE, inertias, 'bo-')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia (Distortion)")

plt.subplot(1, 2, 2)
plt.plot(K_RANGE, silhouettes, 'go-')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Silhouette")

plt.tight_layout()
plt.savefig("data100k/kmeans_diagnostics.png")
plt.show()

#  Match the paper setting: k = 8
best_k = 8
print(f"\n Using best_k = {best_k} (from paper setting)")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
user_clusters = kmeans.fit_predict(X)

# Save user-cluster mapping
with open('data100k/user_clusters.pkl', 'wb') as f:
    pickle.dump(user_clusters, f)

print(f" Saved {best_k} user clusters to data100k/user_clusters.pkl")

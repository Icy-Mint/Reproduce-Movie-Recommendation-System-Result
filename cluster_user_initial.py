import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load encoded user features
with open('data100k/encoded_features.pkl', 'rb') as f:
    X_encoded = pickle.load(f)

# Convert to numpy float array if needed
X = np.array(X_encoded, dtype=np.float32)

# Try a range of cluster counts
K_RANGE = range(2, 15)
inertias = []
silhouettes = []

print("Evaluating KMeans clustering...")
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Plot Elbow and Silhouette scores
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

# Choose best k (based on your plots)
best_k = 8  # ← UPDATE based on elbow/silhouette

# Final KMeans clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
user_clusters = kmeans.fit_predict(X)

# Save user-cluster mapping
with open('data100k/user_clusters.pkl', 'wb') as f:
    pickle.dump(user_clusters, f)

print(f"Clustering done — {best_k} clusters saved to user_clusters.pkl")

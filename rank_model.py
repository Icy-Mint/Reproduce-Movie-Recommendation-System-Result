import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

# Load user-cluster mapping
with open('data100k/user_clusters.pkl', 'rb') as f:
    user_clusters = pickle.load(f)

# Load MovieLens 100K ratings
ratings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Drop timestamp and adjust indexing (MovieLens IDs are 1-based)
ratings.drop(columns=['timestamp'], inplace=True)
ratings['user_id'] -= 1
ratings['item_id'] -= 1

# Attach cluster ID to each rating row
ratings['cluster'] = ratings['user_id'].map(lambda uid: user_clusters[uid])

# Step 1: Compute cluster-wise item rating averages
cluster_item_avg = ratings.groupby(['cluster', 'item_id'])['rating'].mean().unstack(fill_value=np.nan)

# Step 2: Compute overall average rating per cluster
cluster_avg = ratings.groupby('cluster')['rating'].mean()

# Step 3: Predict ratings for all (user, item) pairs
num_users = ratings['user_id'].nunique()
num_items = ratings['item_id'].nunique()
pred_matrix = np.zeros((num_users, num_items))

print("Generating predicted ratings...")

for user_id in range(num_users):
    cluster = user_clusters[user_id]
    for item_id in range(num_items):
        if item_id in cluster_item_avg.columns:
            value = cluster_item_avg.loc[cluster, item_id]
            if not np.isnan(value):
                pred_matrix[user_id][item_id] = value
            else:
                # Item unseen in cluster — fallback to cluster-wide avg
                pred_matrix[user_id][item_id] = cluster_avg[cluster]
        else:
            # Item completely unseen — fallback
            pred_matrix[user_id][item_id] = cluster_avg[cluster]

# Save predicted rating matrix
with open('data100k/predicted_ratings.pkl', 'wb') as f:
    pickle.dump(pred_matrix, f)

print("Rating prediction complete — saved to data100k/predicted_ratings.pkl")

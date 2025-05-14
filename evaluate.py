import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load predicted rating matrix
with open('data100k/predicted_ratings.pkl', 'rb') as f:
    pred_matrix = pickle.load(f)

# Load test ratings (e.g., u1.test)
test = pd.read_csv('datasets/ml-100k/u1.test', sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
test.drop(columns=["timestamp"], inplace=True)

# Adjust 1-based indexing
test["user_id"] -= 1
test["item_id"] -= 1

# Filter: only rows where prediction exists
valid_rows = test[(test["user_id"] < pred_matrix.shape[0]) & (test["item_id"] < pred_matrix.shape[1])]

# Compute RMSE
y_true = valid_rows["rating"].values
y_pred = valid_rows.apply(lambda row: pred_matrix[int(row["user_id"]), int(row["item_id"])], axis=1).values
rmse = sqrt(mean_squared_error(y_true, y_pred))

print(f"RMSE on test set: {rmse:.4f}")

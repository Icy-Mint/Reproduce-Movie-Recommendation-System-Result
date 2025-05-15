#!/usr/bin/env python
"""autoencoder.py  – GHRS 5‑layer AE on the full MovieLens‑100K feature set with loss plot."""

import os, random, numpy as np, tensorflow as tf, pickle, pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pathlib

# reproducibility (paper §4.5)
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42);  np.random.seed(42);  tf.random.set_seed(42)

# ---------------- Parameters ----------------
ALPHA_FILE = "data100k/x_train_alpha(0.01).pkl"   # optimum α = 0.01
ENC_OUT    = "data100k/encoded_features.pkl"
LOSS_OUT   = "data100k/loss.pkl"
FIG_OUT    = "Figs/loss_curve.png"

# Ensure output folders exist
pathlib.Path("data100k").mkdir(parents=True, exist_ok=True)
pathlib.Path("Figs").mkdir(parents=True, exist_ok=True)

# ---------------- Load Data ----------------
X = pd.read_pickle(ALPHA_FILE).astype(np.float32).to_numpy()

# ---------------- Define Autoencoder ----------------
inp = Input((X.shape[1],))
h1  = Dense(64, activation="relu", kernel_regularizer=l1_l2(1e-3))(inp)
h2  = Dense(32, activation="relu", kernel_regularizer=l1_l2(1e-3))(h1)
code= Dense(4 , activation="relu")(h2)               # 4‑dim bottleneck
h3  = Dense(32, activation="relu")(code)
h4  = Dense(64, activation="relu")(h3)
out = Dense(X.shape[1], activation="sigmoid")(h4)

auto = Model(inp, out)
auto.compile(Adam(1e-3), loss="mse")

# ---------------- Train ----------------
history = auto.fit(
    X, X,
    epochs=100,
    batch_size=64,
    shuffle=True,
    validation_split=.2,
    verbose=2
).history

# ---------------- Save Encoded Features ----------------
encoded = Model(inp, code).predict(X)
pickle.dump(encoded.astype(np.float32), open(ENC_OUT, "wb"))
print("✓ encoded_features.pkl saved →", ENC_OUT)

# ---------------- Save Training History ----------------
pickle.dump(history, open(LOSS_OUT, "wb"))
print("✓ loss history saved →", LOSS_OUT)

# ---------------- Plot Loss Curve ----------------
plt.figure(figsize=(8, 5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_OUT)
plt.close()
print("✓ loss curve saved →", FIG_OUT)

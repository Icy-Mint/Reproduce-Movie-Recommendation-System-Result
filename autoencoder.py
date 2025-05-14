import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

# Load combined features
with open('data100k/x_train_alpha(0.01).pkl', 'rb') as f:
    X = pickle.load(f)

# If X is a DataFrame, drop non-numeric columns
if isinstance(X, pd.DataFrame):
    X = X.select_dtypes(include=[np.number])
    X = X.to_numpy(dtype=np.float32)
elif isinstance(X, list):
    X = np.array(X, dtype=np.float32)

# Autoencoder configuration
input_dim = X.shape[1]
encoding_dim = 4  # latent dimension (can be tuned)

input_layer = Input(shape=(input_dim,))
hidden1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(input_layer)
hidden2 = Dense(32, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(hidden1)
encoded = Dense(encoding_dim, activation='relu')(hidden2)
hidden3 = Dense(32, activation='relu')(encoded)
hidden4 = Dense(64, activation='relu')(hidden3)
decoded = Dense(input_dim, activation='sigmoid')(hidden4)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train autoencoder
autoencoder.fit(X, X, epochs=100, batch_size=64, shuffle=True, validation_split=0.2)

# Encode features
X_encoded = encoder.predict(X)

# Save encoded features
with open('data100k/encoded_features.pkl', 'wb') as f:
    pickle.dump(X_encoded, f)

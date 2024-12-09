import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from ActivationFunction import ActivationFunctionFactory, ApproximationType
import matplotlib.pyplot as plt

# Generate synthetic data with two Gaussian clusters per class
np.random.seed(42)
n_samples = 1000
n_features = 4

# Class 0: Two Gaussian clusters
X0_cluster1 = np.random.normal(loc=-2, scale=0.5, size=(n_samples//4, n_features))
X0_cluster2 = np.random.normal(loc=2, scale=0.5, size=(n_samples//4, n_features))
X0 = np.vstack([X0_cluster1, X0_cluster2])

# Class 1: Two Gaussian clusters
X1_cluster1 = np.random.normal(loc=0, scale=0.5, size=(n_samples//2, n_features))
y1 = np.ones(n_samples//2)

# Combine data
X_train = np.vstack([X0, X1_cluster1])
y_train = np.hstack([np.zeros(n_samples//2), y1])

# Shuffle the data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

# Convert to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

# Visualize first two dimensions
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train[:,0]==1, 0], X_train[y_train[:,0]==1, 1], 
           label='Class 0', alpha=0.6)
plt.scatter(X_train[y_train[:,1]==1, 0], X_train[y_train[:,1]==1, 1], 
           label='Class 1', alpha=0.6)
plt.title('First Two Dimensions of Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Define a simple model
inputs = Input(shape=(4,))
x = Dense(8)(inputs)
x = Activation('relu')(x)
x = Dense(2)(x)
outputs = Activation('softmax')(x)
model = Model(inputs, outputs)

# Compile and train the original model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training the original model:")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Save the original model's weights for comparison
original_weights = model.get_weights()

# Create polynomial approximation of ReLU
factory = ActivationFunctionFactory(
    base_activation=tf.keras.activations.relu,
    degree=5,
    approximation_type=ApproximationType.CHEBYSHEV
)
chebyshev_activation = factory.create()

# Replace activation with polynomial approximation
def replace_activation(model, new_activation):
    for layer in model.layers:
        if isinstance(layer, Activation) and layer.activation == tf.keras.activations.relu:
            layer.activation = new_activation

replace_activation(model, chebyshev_activation)

# Compile and train the modified model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nTraining the modified model:")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
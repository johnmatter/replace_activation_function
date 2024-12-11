import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

from ActivationFunction import ActivationFunctionFactory, ApproximationType
from ModelWrapper import ModelWrapper

def create_simple_model(input_shape, regularizer=None):
    model = tf.keras.Sequential()
    
    # Add Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Add BatchNormalization at the start
    model.add(tf.keras.layers.BatchNormalization())
    
    # First Dense layer - make it wider
    if regularizer is not None:
        model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizer))
    else:
        model.add(tf.keras.layers.Dense(128))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Second Dense layer
    if regularizer is not None:
        model.add(tf.keras.layers.Dense(64, kernel_regularizer=regularizer))
    else:
        model.add(tf.keras.layers.Dense(64))
    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    return model

# Generate synthetic data with two Gaussian clusters per class
np.random.seed(42)
n_samples = 10000
n_features = 4

# Class 0: Two Gaussian clusters
X0_cluster1 = np.random.normal(loc=-5, scale=0.5, size=(n_samples//4, n_features))
X0_cluster2 = np.random.normal(loc=5, scale=0.5, size=(n_samples//4, n_features))
X0 = np.vstack([X0_cluster1, X0_cluster2])

# Class 1: One Gaussian cluster
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

# Define the regularizer
l2_regularizer = regularizers.l2(0.001)

# Initialize original model
print("Loading original model...")
original_model = ModelWrapper(
    model_name="simple_model",
    model_type="custom",
    model_class=create_simple_model,
    config_path="training_config.json",
    is_huggingface=False,
    input_shape=(4,),
    debug={
        'print_activation_replacement_debug': True,
        'print_network_split_debug': True
    },
    regularizer=l2_regularizer  # Pass the regularizer here
)
original_model.create_base_model()

# Compile the original model
original_model.model.compile(
    optimizer=original_model.config['training']['optimizer'],
    loss=original_model.config['training']['loss'],
    metrics=original_model.config['training']['metrics']
)

# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

# Train original model
print("Training original model...")
try:
    original_history = original_model.model.fit(
        X_train, y_train,
        epochs=original_model.config['training']['epochs'],
        batch_size=original_model.config['training']['batch_size'],
        validation_split=original_model.config['training']['validation_split'],
        callbacks=[early_stopping]
    )
except KeyboardInterrupt:
    print("Training interrupted by user. Saving current model weights...")
    original_model.model.save_weights("original_model_weights.h5")

# Initialize modified model as a deep copy of the trained original model
print("\nCreating and training modified model...")
modified_model = original_model.copy()

# Create activation function approximation
factory = ActivationFunctionFactory(
    base_activation=tf.keras.activations.relu,
    degree=7,
    approximation_type=ApproximationType.CHEBYSHEV
)
chebyshev_activation = factory.create()

# Build the copied model by passing a dummy input
dummy_input = tf.random.normal((1,) + modified_model.input_shape)
_ = modified_model.model(dummy_input)

# Set activation function first, then split the layers. This must be done in this order for now.
modified_model.set_activation_function(chebyshev_activation)
modified_model.model = modified_model.split_activation_layers()

# Retrain the modified model
print("Retraining the modified model using existing retraining strategies...")
try:
    modified_history = modified_model.retrain(
        X_train, 
        y_train
    )
except KeyboardInterrupt:
    print("Retraining interrupted by user. Saving current model weights...")
    modified_model.model.save_weights("modified_model_weights.h5")

# After generating X_train
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Generate the comparison report
ModelWrapper.generate_comparison_report(
    original_model=original_model,
    modified_model=modified_model,
    X_test=X_train,
    y_test=y_train,
    original_history=original_history,
    modified_history=modified_history,
    report_name="model_comparison_report.pdf"
)
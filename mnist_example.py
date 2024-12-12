import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

from ModelWrapper import ModelWrapper
from ActivationFunction import ActivationFunctionFactory, ApproximationType

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Normalize data
epsilon = 1e-8
X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + epsilon)
X_test = (X_test - np.mean(X_test, axis=0)) / (np.std(X_test, axis=0) + epsilon)

# Define simple CNN model
def create_mnist_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Initialize ModelWrapper
mnist_model = ModelWrapper(
    model_name="mnist_cnn",
    model_type="classification",
    model_class=create_mnist_model,
    input_shape=(28, 28, 1)
)
mnist_model.create_base_model()

# Configure for MNIST (10 classes)
mnist_model.configure_for_classification(num_classes=10)
mnist_model.validate_model_output(expected_classes=10)

# Compile the model
mnist_model.model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

# Train the model
history = mnist_model.model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=128,
    callbacks=[early_stopping]
)

# Generate analysis report
mnist_model.generate_analysis_report(
    X_test=X_test,
    y_test=y_test,
    history=history,
    report_name="mnist_analysis_report.pdf"
) 

# Initialize modified model as a deep copy of the trained original model
print("\nCreating and training modified model...")
modified_model = mnist_model.copy()

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

# After splitting layers, verify output shape
test_output = modified_model.model.predict(dummy_input)
if test_output.shape[-1] != 10:
    raise ValueError(f"Model output shape changed! Expected 10 classes, got {test_output.shape[-1]}")

# Validate modified model maintains multi-class configuration
modified_model.configure_for_classification(num_classes=10)
modified_model.validate_model_output(expected_classes=10)

# Plot model summaries for comparison
print("Original Model Summary:")
mnist_model.model.summary()

print("\nModified Model Summary:")
modified_model.model.summary()

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

# Normalize X_train if needed (ensuring consistency)
X_train_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Generate the comparison report
ModelWrapper.generate_comparison_report(
    original_model=mnist_model,
    modified_model=modified_model,
    X_test=X_test,  # Ensure X_test is correctly used
    y_test=y_test,
    original_history=history,
    modified_history=modified_history,
    report_name="mnist_comparison_report.pdf"
)
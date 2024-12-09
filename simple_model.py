import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from ActivationFunction import ActivationFunctionFactory, ApproximationType
import matplotlib.pyplot as plt
from ModelWrapper import ModelWrapper

def generate_analysis_report(
    original_model: ModelWrapper,
    modified_model: ModelWrapper,
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: str = "analysis_report.pdf"
) -> None:
    """
    Generate a PDF report comparing original and modified models.
    
    Args:
        original_model: Original ModelWrapper instance
        modified_model: Modified ModelWrapper instance with polynomial approximations
        X_train: Training data
        y_train: Training labels
        output_path: Path to save the PDF report
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(output_path) as pdf:
        # Page 1: Data Distribution
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[y_train[:,0]==1, 0], X_train[y_train[:,0]==1, 1], 
                   label='Class 0', alpha=0.6)
        plt.scatter(X_train[y_train[:,1]==1, 0], X_train[y_train[:,1]==1, 1], 
                   label='Class 1', alpha=0.6)
        plt.title('Training Data Distribution\n(First Two Dimensions)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # Page 2: Prediction Comparisons
        num_samples_to_plot = 100
        test_samples = X_train[:num_samples_to_plot]
        original_preds = original_model.model.predict(test_samples)
        modified_preds = modified_model.model.predict(test_samples)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.scatter(original_preds[:, 0], modified_preds[:, 0], 
                   alpha=0.5, label='Class 0')
        plt.scatter(original_preds[:, 1], modified_preds[:, 1], 
                   alpha=0.5, label='Class 1')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
        plt.xlabel('Original Model Probability')
        plt.ylabel('Modified Model Probability')
        plt.title('Prediction Probability Comparison')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        differences = np.abs(original_preds - modified_preds)
        plt.hist(differences.flatten(), bins=50, alpha=0.7)
        plt.xlabel('Absolute Difference in Predictions')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Differences')
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 3: Model Summary
        plt.figure(figsize=(8, 10))
        plt.text(0.1, 0.9, 'Model Comparison Summary', fontsize=14, fontweight='bold')
        
        # Calculate summary statistics
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        accuracy_agreement = np.mean(
            original_preds.argmax(axis=1) == modified_preds.argmax(axis=1)
        )
        
        summary_text = (
            f"Number of samples analyzed: {num_samples_to_plot}\n\n"
            f"Mean prediction difference: {mean_diff:.4f}\n"
            f"Maximum prediction difference: {max_diff:.4f}\n"
            f"Prediction agreement rate: {accuracy_agreement:.2%}\n\n"
            f"Original Model Architecture:\n"
            f"{original_model.model.summary()}\n\n"
            f"Modified Model Architecture:\n"
            f"{modified_model.model.summary()}"
        )
        
        plt.text(0.1, 0.8, summary_text, fontsize=10, 
                verticalalignment='top', transform=plt.gca().transAxes)
        plt.axis('off')
        pdf.savefig()
        plt.close()

# Generate synthetic data with two Gaussian clusters per class
np.random.seed(42)
n_samples = 100000
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

def create_simple_model():
    inputs = Input(shape=(4,))
    x = Dense(8)(inputs)
    x = Activation('relu')(x)
    x = Dense(2)(x)
    outputs = Activation('softmax')(x)
    return Model(inputs, outputs)

# Initialize original model
print("Loading original model...")
original_model = ModelWrapper(
    model_name="simple_model",
    model_type="custom",
    model_class=create_simple_model,  # Pass the function directly
    is_huggingface=False,
    input_shape=(4,),
    debug={
        'print_activation_replacement_debug': True
    }
)
original_model.create_base_model()

# Train original model
original_model.model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
original_model.model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Initialize modified model
print("\nCreating and training modified model...")
modified_model = ModelWrapper(
    model_name="simple_model",
    model_type="simple",
    model_class=create_simple_model,
    is_huggingface=False,
    input_shape=(4,),
    debug={
        'print_activation_replacement_debug': True,
        'print_network_split_debug': True
    }
)
modified_model.create_base_model()

# Create activation function approximation
factory = ActivationFunctionFactory(
    base_activation=tf.keras.activations.relu,
    degree=5,
    approximation_type=ApproximationType.CHEBYSHEV
)
chebyshev_activation = factory.create()

# Split activation layers and set activation function
modified_model.model = modified_model.split_activation_layers()
modified_model.set_activation_function(chebyshev_activation)

# Train modified model
modified_model.model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modified_model.model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Generate analysis report
generate_analysis_report(
    original_model=original_model,
    modified_model=modified_model,
    X_train=X_train,
    y_train=y_train,
    output_path="model_analysis_report.pdf"
)
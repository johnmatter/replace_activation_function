import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.metrics import precision_recall_fscore_support

from ActivationFunction import ActivationFunctionFactory, ApproximationType
from ModelWrapper import ModelWrapper

def generate_analysis_report(
    original_model: ModelWrapper,
    modified_model: ModelWrapper,
    X_train: np.ndarray,
    y_train: np.ndarray,
    original_history: tf.keras.callbacks.History,
    modified_history: dict,
    output_path: str = "analysis_report.pdf"
) -> None:
    """
    Generate a PDF report comparing original and modified models.

    Args:
        original_model: Original ModelWrapper instance
        modified_model: Modified ModelWrapper instance with polynomial approximations
        X_train: Training data
        y_train: Training labels
        original_history: Training history of the original model
        modified_history: Training history of the modified model
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

        # Page 2: Training Metrics - Original Model
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        epochs = range(1, len(original_history.history['loss']) + 1)
        
        # Loss subplot
        ax1.plot(epochs, original_history.history['loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, original_history.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Original Model Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy subplot
        ax2.plot(epochs, original_history.history['accuracy'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, original_history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        ax2.set_title('Original Model Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 3 onwards: Modified Model Training Metrics
        retrain_type = modified_model.config['retraining']['retraining_type']

        if retrain_type == 'all':
            # Handle 'all' retraining strategy
            for phase in ['train_before_activation', 'train_after_activation']:
                if phase in modified_history:
                    hist = modified_history[phase]
                    plt.figure(figsize=(12, 6))
                    epochs = range(1, len(hist['loss']) + 1)
                    plt.plot(epochs, hist['loss'], label='Training Loss')
                    plt.plot(epochs, hist['val_loss'], label='Validation Loss')
                    plt.plot(epochs, hist['accuracy'], label='Training Accuracy')
                    plt.plot(epochs, hist['val_accuracy'], label='Validation Accuracy')
                    plt.title(f'Modified Model Training Metrics ({phase.replace("_", " ").title()})')
                    plt.xlabel('Epoch')
                    plt.ylabel('Metric')
                    plt.legend()
                    plt.grid(True)
                    pdf.savefig()
                    plt.close()

        elif retrain_type == 'iterative':
            # First, find the min/max values across all layers for consistent scaling
            loss_min, loss_max = float('inf'), float('-inf')
            acc_min, acc_max = float('inf'), float('-inf')
            
            for entry in modified_history['iterative']:
                hist = entry['history']
                loss_min = min(loss_min, min(hist['loss']), min(hist['val_loss']))
                loss_max = max(loss_max, max(hist['loss']), max(hist['val_loss']))
                acc_min = min(acc_min, min(hist['accuracy']), min(hist['val_accuracy']))
                acc_max = max(acc_max, max(hist['accuracy']), max(hist['val_accuracy']))
            
            # Add some padding to the ranges
            loss_range = loss_max - loss_min
            acc_range = acc_max - acc_min
            loss_min -= loss_range * 0.1
            loss_max += loss_range * 0.1
            acc_min = max(0, acc_min - acc_range * 0.1)  # Don't go below 0 for accuracy
            acc_max = min(1, acc_max + acc_range * 0.1)  # Don't exceed 1 for accuracy
            
            # Create a figure with subplots for all layers
            n_layers = len(modified_history['iterative'])
            fig, axes = plt.subplots(2, n_layers, figsize=(6*n_layers, 12))
            
            # If there's only one layer, axes needs to be 2D
            if n_layers == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, entry in enumerate(modified_history['iterative']):
                hist = entry['history']
                layer_idx = entry['layer']
                epochs = range(1, len(hist['loss']) + 1)
                
                # Loss subplot
                ax_loss = axes[0, idx]
                ax_loss.plot(epochs, hist['loss'], 'b-', label='Training Loss')
                ax_loss.plot(epochs, hist['val_loss'], 'r-', label='Validation Loss')
                ax_loss.set_title(f'Layer {layer_idx + 1} Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_ylim(loss_min, loss_max)
                ax_loss.legend()
                ax_loss.grid(True)
                
                # Accuracy subplot
                ax_acc = axes[1, idx]
                ax_acc.plot(epochs, hist['accuracy'], 'b-', label='Training Accuracy')
                ax_acc.plot(epochs, hist['val_accuracy'], 'r-', label='Validation Accuracy')
                ax_acc.set_title(f'Layer {layer_idx + 1} Accuracy')
                ax_acc.set_xlabel('Epoch')
                ax_acc.set_ylabel('Accuracy')
                ax_acc.set_ylim(acc_min, acc_max)
                ax_acc.legend()
                ax_acc.grid(True)
            
            plt.suptitle('Modified Model Training Metrics by Layer', fontsize=16, y=1.02)
            plt.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        elif retrain_type == 'batched':
            # Handle 'batched' retraining strategy
            for entry in modified_history['batched']:
                hist = entry['history']
                batch_idx = entry['batch']
                phase = entry['phase']
                plt.figure(figsize=(12, 6))
                epochs = range(1, len(hist['loss']) + 1)
                plt.plot(epochs, hist['loss'], label='Training Loss')
                plt.plot(epochs, hist['val_loss'], label='Validation Loss')
                plt.plot(epochs, hist['accuracy'], label='Training Accuracy')
                plt.plot(epochs, hist['val_accuracy'], label='Validation Accuracy')
                plt.title(f'Batched Retraining (Batch {batch_idx + 1}, {phase.replace("_", " ").title()})')
                plt.xlabel('Epoch')
                plt.ylabel('Metric')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.close()

        # Page 4: Prediction Comparisons
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

        # Page 5: Model Summary
        from io import StringIO
        import sys

        # Capture model summaries
        def capture_model_summary(model):
            stream = StringIO()
            sys.stdout = stream
            model.summary()
            sys.stdout = sys.__stdout__
            summary_string = stream.getvalue()
            stream.close()
            return summary_string

        original_summary = capture_model_summary(original_model.model)
        modified_summary = capture_model_summary(modified_model.model)

        # Display model summaries and metrics
        plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.title('Model Comparison Summary', fontsize=14, fontweight='bold')

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
            f"Original Model Architecture:\n{original_summary}\n\n"
            f"Modified Model Architecture:\n{modified_summary}"
        )

        plt.text(0.01, 0.99, summary_text, fontsize=10, va='top', ha='left')
        pdf.savefig()
        plt.close()

        # Calculate predictions for metrics
        original_preds = original_model.model.predict(X_train)
        modified_preds = modified_model.model.predict(X_train)
        
        # Calculate metrics
        orig_precision, orig_recall, orig_f1, _ = precision_recall_fscore_support(
            y_train.argmax(axis=1),
            original_preds.argmax(axis=1),
            average='binary'
        )
        
        mod_precision, mod_recall, mod_f1, _ = precision_recall_fscore_support(
            y_train.argmax(axis=1),
            modified_preds.argmax(axis=1),
            average='binary'
        )

        # Plot comparison metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        metrics = {
            'Precision': (orig_precision, mod_precision),
            'Recall': (orig_recall, mod_recall),
            'F1 Score': (orig_f1, mod_f1),
            'Accuracy': (
                np.mean(y_train.argmax(axis=1) == original_preds.argmax(axis=1)),
                np.mean(y_train.argmax(axis=1) == modified_preds.argmax(axis=1))
            )
        }

        for (metric_name, (orig_val, mod_val)), ax in zip(metrics.items(), axes.flatten()):
            x = ['Original Model', 'Modified Model']
            y = [orig_val, mod_val]
            ax.bar(x, y)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_ylim(0, 1)
            for i, v in enumerate(y):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"Analysis report saved to {output_path}")

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

# Generate analysis report
generate_analysis_report(
    original_model=original_model,
    modified_model=modified_model,
    X_train=X_train,
    y_train=y_train,
    original_history=original_history,
    modified_history=modified_history,
    output_path="model_analysis_report.pdf"
)
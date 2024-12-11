from enum import Enum
from typing import Any, Dict, List, Union, Optional, Callable, Tuple
import json
import os
from enum import Enum
from typing import Any, Dict, List, Union, Optional, Callable, Tuple
import numpy as np
import tensorflow as tf
import warnings
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoImageProcessor,
    ResNetForImageClassification
)
from PIL import Image
import time
import copy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from PredictionDecoder import PredictionDecoder

# Keras includes the following models:
# - ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
# - VGG16, VGG19
# - InceptionV3
# - InceptionResNetV2
# - MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
# - DenseNet121, DenseNet169, DenseNet201
# - NASNetMobile, NASNetLarge
# - EfficientNetB0 through EfficientNetB7
# - Xception
# - ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge

class RetrainType(Enum):
    """Enum for different types of retraining"""
    ALL = "all"
    BATCHED = "batched"
    ITERATIVE = "iterative"

class ModelWrapper:
    def __init__(
        self, 
        model_name: str = "model", 
        model_type: str = "classification", 
        debug: Dict[str, bool] = None, 
        model_class: Optional[Any] = None, 
        processor_class: Optional[Any] = None, 
        is_huggingface: bool = False,
        input_shape: Tuple[int, ...] = (28, 28, 1),
        config_path: Optional[str] = None,
        **model_kwargs
    ) -> None:

        # # Enable eager execution at the start
        # tf.config.run_functions_eagerly(True)
        
        # Initialize model-specific attributes
        self.model_name = model_name
        self.model_type = model_type
        self.debug = debug or {}
        self.model = None
        self.processor = None
        self.model_class = model_class
        self.processor_class = processor_class
        self.is_huggingface = is_huggingface
        self.input_shape = input_shape
        self.config = self._load_config(config_path)
        self.activation_function = None
        self.activation_function_to_replace = self.config.get('activation_type_to_replace')
        self.activation_layer_pairs = []  # List of (original_layer, activation_layer) tuples
        self.model_kwargs = model_kwargs  # Store additional kwargs

        # Initialize compilation settings
        self.base_settings = {
            'loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            'metrics': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'jit_compile': False
        }
        
        # Initialize optimizer settings
        self.current_compile_settings = None
        self.current_optimizer = None
        
        # Initialize debug settings
        self.initialize_debug()

        # If config contains activation type to replace, set it
        if 'activation_type_to_replace' in self.config:
            self.activation_function_to_replace = self.config['activation_type_to_replace']

        if self.activation_function_to_replace is None:
            raise ValueError("activation_type_to_replace must be specified in the config.")

    def save(self, model_path: str) -> None:
        """Save the model"""
        self.model.save(model_path)

    def print_model_summary(self) -> None:
        """Print the model summary"""
        self.model.summary()

    def initialize_debug(self) -> None:
        if self.debug is None:
            self.debug = {
                'print_network_split_debug': False,
                'print_activation_replacement_debug': True
            }
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file"""
        default_path = os.path.join(os.path.dirname(__file__), 'training_config.json')
        config_path = config_path or default_path
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}, using default settings.")
            return {}

    def set_activation_function(self, activation_function):
        """Set the activation function to use for replacement."""
        self.activation_function = activation_function
        return self
        
    def create_base_model(self) -> 'ModelWrapper':
        """Initialize the model and processor based on model type"""
        if not self.is_huggingface:
            if self.model_class is not None:
                if callable(self.model_class) and not isinstance(self.model_class, type):
                    # Handle custom model function
                    self.model = self.model_class(self.input_shape, **self.model_kwargs)
                else:
                    # Handle built-in Keras models
                    if self.model_type == "image":
                        self.model = self.model_class(
                            weights='imagenet',
                            input_shape=self.input_shape,
                            include_top=True
                        )
                    else:
                        self.model = self.model_class()

                if self.processor_class is not None:
                    self.processor = self.processor_class
            else:
                raise ValueError("Model class must be provided")
        else:
            # Handle HuggingFace models if needed
            pass

        print("Model created successfully.")
        return self

    def split_activation_layers(self) -> tf.keras.Model:
        """Split activation layers from their parent layers."""
        if self.activation_function_to_replace is None:
            raise ValueError("activation_function_to_replace must be set in config before splitting layers")
        
        if self.debug.get('print_network_split_debug', False):
            print("\nStarting split_activation_layers")

        # Reset activation_layer_pairs
        self.activation_layer_pairs = []

        # Check if the model is Sequential or Functional
        if isinstance(self.model, tf.keras.Sequential):
            if self.debug.get('print_network_split_debug', False):
                print("Model is Sequential.")

            # Create a new list of layers
            new_layers = []

            for layer in self.model.layers:
                # Get layer config
                config = layer.get_config()

                # Check if the layer has 'activation' in its config
                has_activation_in_config = (
                    'activation' in config and
                    config['activation'] is not None and
                    config['activation'] != 'linear'
                )

                if has_activation_in_config and config['activation'] == self.config['activation_type_to_replace']:
                    # Remove activation from config
                    original_activation = config['activation']
                    config['activation'] = 'linear'

                    if self.debug.get('print_activation_replacement_debug', False):
                        print(f"Splitting activation in layer {layer.name}")

                    # Clone layer without activation
                    new_layer = layer.__class__.from_config(config)

                    # Build and set weights only if the layer has weights
                    if layer.get_weights() and hasattr(layer, 'input_shape'):
                        new_layer.build(layer.input_shape)
                        new_layer.set_weights(layer.get_weights())
                    # Another approach might be:
                    # weight_layers = (
                        # tf.keras.layers.Dense,
                        # tf.keras.layers.Conv2D,
                        # tf.keras.layers.BatchNormalization,
                        # Add other layer types as needed
                    # )
                    # if isinstance(layer, weight_layers):
                        # new_layer.build(layer.input_shape)
                        # new_layer.set_weights(layer.get_weights())

                    # Append the new layer
                    new_layers.append(new_layer)

                    # Create new activation layer
                    activation_layer = tf.keras.layers.Activation(
                        activation=original_activation,
                        name=f"{layer.name}_activation"
                    )
                    new_layers.append(activation_layer)

                    # Save the pair for replacement during retraining
                    self.activation_layer_pairs.append((new_layer, activation_layer))
                else:
                    # Append the layer as-is
                    new_layers.append(layer)

            # Create a new Sequential model
            new_model = tf.keras.Sequential(new_layers, name=self.model.name)

            # Update the model reference
            self.model = new_model

        else:
            if self.debug.get('print_network_split_debug', False):
                print("Model is Functional.")

            # Define the clone function
            def clone_function(layer):
                # Get layer config
                config = layer.get_config()

                # Check if the layer has an 'activation' in its config
                has_activation_in_config = (
                    'activation' in config and
                    config['activation'] is not None and
                    config['activation'] != 'linear'
                )

                if has_activation_in_config and config['activation'] == self.config['activation_type_to_replace']:
                    # Remove activation from config
                    original_activation = config['activation']
                    config['activation'] = 'linear'

                    if self.debug.get('print_activation_replacement_debug', False):
                        print(f"Splitting activation in layer {layer.name}")

                    # Clone the layer without activation
                    new_layer = layer.__class__.from_config(config)

                    # Return a function that applies the layer and then adds the new activation layer
                    def apply_new_layer(inputs):
                        x = new_layer(inputs)
                        activation_layer = tf.keras.layers.Activation(
                            activation=original_activation,
                            name=f"{layer.name}_activation"
                        )
                        x = activation_layer(x)
                        # Save the pair for replacement during retraining
                        self.activation_layer_pairs.append((new_layer, activation_layer))
                        return x

                    return apply_new_layer

                # Return the layer as-is
                return layer

            # Clone the model using the custom clone_function
            new_model = tf.keras.models.clone_model(
                self.model,
                clone_function=clone_function
            )

            # Copy weights from the original model to the new model
            new_model.set_weights(self.model.get_weights())

            # Update the model reference
            self.model = new_model

        if self.debug.get('print_network_split_debug', False):
            print("Finished split_activation_layers")

        return self.model
        
    def replace_all_activations(self, original_activation_function: Callable, replacement_activation_function: Callable) -> 'ModelWrapper':
        replacements_made = 0
        skipped_replacements = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                if layer.activation.__name__ == original_activation_function.__name__:   
                    if self.debug['print_activation_replacement_debug']:
                        print(f"Replacing activation {original_activation_function.__name__} with {replacement_activation_function.poly.dump()}")
                    layer.activation = replacement_activation_function
                    replacements_made += 1
                else:
                    skipped_replacements += 1
        
        if replacements_made == 0:
            print("Warning: No activation functions were replaced!")
        else:
            print(f"Replaced {replacements_made} activation functions")

        return self

    def replace_activation(self, activation_layer, new_activation):
        """Replace the activation function of an Activation layer."""
        if isinstance(activation_layer, tf.keras.layers.Activation):
            if self.debug.get('print_activation_replacement_debug', False):
                print(f"Replacing activation in layer {activation_layer.name}")
            activation_layer.activation = new_activation

            # # Add L2 regularization to kernel if available
            # if hasattr(layer, 'kernel_regularizer'):
            #     layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    def preprocess(self, inputs):
        if self.is_huggingface:
            return self.processor(inputs, return_tensors="tf")
        else:
            # For ResNet50 and similar models
            if isinstance(inputs, Image.Image):
                # Convert PIL Image to numpy array
                inputs = np.array(inputs)
            # Expand dimensions to create batch of size 1
            inputs = np.expand_dims(inputs, axis=0)
            return self.processor(inputs)

    def predict(self, inputs: Union[str, List[str], np.ndarray, List[np.ndarray]]) -> Dict[str, Any]:
        """Make predictions on inputs"""
        # Ensure model and processor are loaded
        if self.model is None or self.processor is None:
            self.create_base_model()
            
        # Preprocess the input
        processed_inputs = self.preprocess(inputs)
        
        # Get predictions
        outputs = self.model(processed_inputs)
        
        # Create model info dictionary
        model_info = {
            'is_huggingface': self.is_huggingface,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'top_k': self.config['prediction_decoder']['top_k']
        }
        
        # Add HuggingFace specific info if needed
        if self.is_huggingface:
            model_info['id2label'] = self.model.config.id2label
        
        # Use PredictionDecoder to handle the outputs
        return PredictionDecoder.decode(outputs, model_info)

    def get_layer_before(self, activation_layer: tf.keras.layers.Layer) -> Optional[tf.keras.layers.Layer]:
        """Get the layer that comes before the given activation layer"""
        # Raise an error if the layer is the first layer
        if self.model.layers.index(activation_layer) == 0:
            raise ValueError("The specified layer is the first layer in the model")
        layers = self.model.layers
        for i, layer in enumerate(layers):
            if layer is activation_layer and i > 0:
                return layers[i - 1]
        return None

    def get_layer_after(self, activation_layer: tf.keras.layers.Layer) -> Optional[tf.keras.layers.Layer]:
        """Get the layer that comes after the given activation layer"""
        layers = self.model.layers
        for i, layer in enumerate(layers):
            if layer is activation_layer and i < len(layers) - 1:
                return layers[i + 1]
        return None
    
    def get_trainable_layer_before(self, target_layer):
        """Find the first trainable layer before the target layer."""
        target_index = self.model.layers.index(target_layer)
        for layer in reversed(self.model.layers[:target_index]):
            if self.is_trainable_layer(layer):
                return layer
        return None

    def get_trainable_layer_after(self, target_layer):
        """Find the first trainable layer after the target layer."""
        target_index = self.model.layers.index(target_layer)
        for layer in self.model.layers[target_index + 1:]:
            if self.is_trainable_layer(layer):
                return layer
        return None

    def is_trainable_layer(self, layer):
        """Determine if a layer is trainable."""
        return hasattr(layer, 'trainable') and isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D))

    def add_layer_before(self, layer: tf.keras.layers.Layer, new_layer: tf.keras.layers.Layer) -> 'ModelWrapper':
        """Add a layer before the specified layer"""
        self.model.layers.insert(self.model.layers.index(layer), new_layer)
        return self

    def batch_renormalize_before_activation(self, activation_layer: tf.keras.layers.Activation) -> 'ModelWrapper':
        """Batch normalize the layer before the specified activation layer"""
        layer_before = self.get_layer_before(activation_layer)
        layer_before.add_before(tf.keras.layers.BatchNormalization())
        return self

    def retrain(self, train_data: np.ndarray, train_labels: np.ndarray) -> dict:
        """
        Retrain the model using the specified strategy from config.

        Args:
            train_data: Training data.
            train_labels: Training labels.

        Returns:
            A dictionary containing training histories.
        """
        from tensorflow.keras.callbacks import History

        # Validate activation function settings
        self._validate_activation_replacement()

        # Retrieve retraining configuration
        retraining_config = self.config.get('retraining', {})
        retraining_type = retraining_config.get('retraining_type')

        if retraining_type not in ['all', 'iterative', 'batched']:
            raise ValueError(f"Unsupported retraining_type: {retraining_type}")

        batch_size = retraining_config.get('batch_size', 32)
        activation_replacement_batch_size = retraining_config.get('activation_replacement_batch_size', 1)
        epochs = retraining_config.get('epochs', 5)

        # Prepare datasets
        validation_split = self.config['training'].get('validation_split', 0.1)
        split_index = int(len(train_data) * (1 - validation_split))
        x_train, x_val = train_data[:split_index], train_data[split_index:]
        y_train, y_val = train_labels[:split_index], train_labels[split_index:]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        history = {}

        if retraining_type == 'all':
            history = self._retrain_all(train_dataset, val_dataset, epochs)
        elif retraining_type == 'iterative':
            history = self._retrain_iterative(train_dataset, val_dataset, epochs)
        elif retraining_type == 'batched':
            history = self._retrain_batched(train_dataset, val_dataset, activation_replacement_batch_size, epochs)

        return history

    def _retrain_all(self, train_dataset, val_dataset, epochs):
        """Retrain all layers at once."""
        history = {}
        # Assuming 'all_mode' parameters are now part of the unified config
        # Retrieve any specific settings if needed
        config = self.config['retraining']
        
        # Optional: Train before activation replacement
        if config.get('train_before_activation', False):
            print("Training before activation replacement...")
            self.model.compile(
                optimizer=self.get_optimizer(),
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )
            hist_before = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )
            history['train_before_activation'] = hist_before.history

        # Replace all specified activations
        print("Replacing all specified activation functions...")
        for activation_layer in self.get_activation_layers():
            self.replace_activation(activation_layer, self.activation_function)

        # Optional: Train after activation replacement
        if config.get('train_after_activation', False):
            print("Training after activation replacement...")
            self.reset_optimizer(learning_rate_scale=0.1)  # Assuming this method resets the optimizer appropriately
            self.model.compile(
                optimizer=self.get_optimizer(),
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )
            hist_after = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )
            history['train_after_activation'] = hist_after.history

        return history

    def _retrain_iterative(self, train_dataset, val_dataset, epochs):
        """Iteratively retrain layers."""
        history = {'iterative': []}
        activation_layers = self.get_activation_layers()

        for idx, activation_layer in enumerate(activation_layers):
            if self.debug.get('print_retrain_debug', False):
                print(f"\nProcessing activation layer {idx+1}/{len(activation_layers)}: {activation_layer.name}")

            # Replace activation
            self.replace_activation(activation_layer, self.activation_function)

            # Reset all layers to non-trainable
            for layer in self.model.layers:
                layer.trainable = False

            # Find and set trainable layers
            prev_layer = self.get_trainable_layer_before(activation_layer)
            next_layer = self.get_trainable_layer_after(activation_layer)

            if prev_layer:
                prev_layer.trainable = True
                if self.debug.get('print_retrain_debug', False):
                    print(f"Training layer before activation: {prev_layer.name}")

            if next_layer:
                next_layer.trainable = True
                if self.debug.get('print_retrain_debug', False):
                    print(f"Training layer after activation: {next_layer.name}")

            # Create optimizer with scaled learning rate
            optimizer = self.get_optimizer(learning_rate_scale=0.1)

            # Compile the model
            self.model.compile(
                optimizer=optimizer,
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )

            # Train the model
            print(f"Retraining after replacing activation in layer {activation_layer.name}...")
            hist = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )

            # Record history
            history_entry = {
                'layer': idx,
                'phase': 'post_activation_replacement',
                'history': hist.history
            }
            history['iterative'].append(history_entry)

        return history

    def _retrain_batched(self, train_dataset, val_dataset, activation_replacement_batch_size, epochs):
        """Batched retraining based on activation_replacement_batch_size."""
        history = {'batched': []}
        activation_layers = self.get_activation_layers()
        num_batches = (len(activation_layers) + activation_replacement_batch_size - 1) // activation_replacement_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * activation_replacement_batch_size
            end_idx = min((batch_idx + 1) * activation_replacement_batch_size, len(activation_layers))
            batch_layers = activation_layers[start_idx:end_idx]

            if self.debug.get('print_retrain_debug', False):
                print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

            # Replace activations in this batch
            for activation_layer in batch_layers:
                self.replace_activation(activation_layer, self.activation_function)
                if self.debug.get('print_retrain_debug', False):
                    print(f"Replaced activation in layer: {activation_layer.name}")

            # Reset all layers to non-trainable
            for layer in self.model.layers:
                layer.trainable = False

            # Find and set trainable layers for current batch
            for activation_layer in batch_layers:
                prev_layer = self.get_trainable_layer_before(activation_layer)
                next_layer = self.get_trainable_layer_after(activation_layer)

                if prev_layer:
                    prev_layer.trainable = True
                    if self.debug.get('print_retrain_debug', False):
                        print(f"Training layer before activation: {prev_layer.name}")

                if next_layer:
                    next_layer.trainable = True
                    if self.debug.get('print_retrain_debug', False):
                        print(f"Training layer after activation: {next_layer.name}")

            # Create optimizer with scaled learning rate
            optimizer = self.get_optimizer(learning_rate_scale=0.1)

            # Compile the model
            self.model.compile(
                optimizer=optimizer,
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )

            # Train the model
            print(f"Retraining batch {batch_idx + 1}/{num_batches}...")
            hist = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )

            # Record history
            history_entry = {
                'batch': batch_idx,
                'phase': 'post_activation_replacement',
                'history': hist.history
            }
            history['batched'].append(history_entry)

        return history

    def get_input_layer_name(self) -> str:
        """Get the current input layer name of the model"""
        # Try multiple methods to get the input layer name
        if hasattr(self.model, 'input_names') and self.model.input_names:
            return self.model.input_names[0]
        
        if hasattr(self.model, 'input_layer'):
            if self.model.input_layer is not None:
                return self.model.input_layer.name
        
        if hasattr(self.model, 'input'):
            try:
                # Get the name without any tensor suffixes
                name = self.model.input.name.split(':')[0]
                # If the name contains keras_tensor, use the layer name instead
                if 'keras_tensor' in name:
                    return self.model.layers[0].name
                return name
            except (AttributeError, TypeError):
                pass
        
        # Fallback to first layer name
        return self.model.layers[0].name

    def prepare_data(self, data: np.ndarray, labels: np.ndarray) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Prepare and validate data for training, returning train and validation datasets"""
        training_config = self.config['training']
        validation_split = training_config['validation_split']

        # Convert to tensors if needed
        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        if isinstance(labels, np.ndarray):
            labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Get shapes
        data_shape = tf.shape(data)
        model_input_shape = self.model.input_shape
        
        # Handle different dimension cases
        if len(data.shape) < len(model_input_shape):
            # Add missing dimensions
            for _ in range(len(model_input_shape) - len(data.shape)):
                data = tf.expand_dims(data, axis=0)
        elif len(data.shape) > len(model_input_shape):
            # Remove extra dimensions if they're singleton
            while len(data.shape) > len(model_input_shape):
                if data.shape[0] == 1:
                    data = tf.squeeze(data, axis=0)
                else:
                    raise ValueError(f"Data shape {data.shape} has too many non-singleton dimensions for model input shape {model_input_shape}")
        
        # Validate final shapes
        dynamic_axes = [i for i, dim in enumerate(model_input_shape) if dim is None]
        static_axes = [i for i, dim in enumerate(model_input_shape) if dim is not None]
        
        for axis in static_axes[1:]:  # Skip batch dimension
            if data.shape[axis] != model_input_shape[axis]:
                raise ValueError(f"Dimension mismatch at axis {axis}: expected {model_input_shape[axis]}, got {data.shape[axis]}")

        # Calculate split point for validation
        num_samples = tf.cast(tf.shape(data)[0], tf.float32)
        split_at = tf.cast(num_samples * (1 - validation_split), tf.int32)

        # Create training dataset with proper mapping
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (data[:split_at], labels[:split_at])
        ).cache().batch(
            training_config['batch_size'],
            drop_remainder=True
        ).map(
            lambda x, y: (x, tf.cast(y, tf.float32))
        ).prefetch(tf.data.AUTOTUNE)

        # Create validation dataset with proper mapping
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (data[split_at:], labels[split_at:])
        ).batch(
            training_config['batch_size'],
            drop_remainder=True
        ).map(
            lambda x, y: (x, tf.cast(y, tf.float32))
        ).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def reset_optimizer(self, learning_rate_scale=1.0):
        """
        Reset the optimizer with a scaled learning rate.
        This function assumes that self.current_compile_settings is already set.
        """
        training_config = self.current_compile_settings

        optimizer_name = training_config['optimizer'].lower()

        # Get current learning rate
        current_lr = None
        if hasattr(self.model.optimizer, 'learning_rate'):
            current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        else:
            # Default learning rate based on optimizer
            if optimizer_name == 'adam':
                current_lr = 0.001
            elif optimizer_name == 'sgd':
                current_lr = 0.01
            else:
                current_lr = 0.001  # Default fallback

        # Scale learning rate
        new_lr = current_lr * learning_rate_scale

        if optimizer_name == 'adam':
            self.model.optimizer = tf.keras.optimizers.Adam(learning_rate=new_lr)
        elif optimizer_name == 'sgd':
            self.model.optimizer = tf.keras.optimizers.SGD(learning_rate=new_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_callbacks(self):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ]

    def _reset_trainable_layers(self):
        """Reset all layers to non-trainable state"""
        for layer in self.model.layers:
            layer.trainable = False

    def _fit(self, *args, **kwargs):
        """Wrapper around model.fit with safety checks"""
        # Count trainable parameters
        trainable_params = sum([
            tf.size(var).numpy() 
            for var in self.model.trainable_variables
        ])
        
        if trainable_params == 0:
            raise ValueError(
                "Model has 0 trainable parameters. "
                f"Check that layers are properly set to trainable before training. "
                f"Current trainable layers: {[layer.name for layer in self.model.layers if layer.trainable]}"
            )
        
        # Print trainable parameter count and layers for debugging
        print(f"\nTraining with {trainable_params:,} trainable parameters")
        print("Trainable layers:")
        for layer in self.model.layers:
            if layer.trainable:
                print(f"- {layer.name}: {sum(tf.size(var).numpy() for var in layer.trainable_variables):,} parameters")
        
        return self.model.fit(*args, **kwargs)

    def get_activation_layers(self) -> List[tf.keras.layers.Layer]:
        """Get a list of activation layers that are to be replaced."""
        self._validate_activation_replacement()
        
        activation_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                if layer.activation.__name__ == self.activation_function_to_replace:
                    activation_layers.append(layer)
        return activation_layers

    def get_optimizer(self, learning_rate_scale=1.0):
        """Create optimizer based on config with scaled learning rate."""
        training_config = self.config['training']
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        base_learning_rate = training_config.get('learning_rate', 0.001)

        # Scale the learning rate
        new_lr = base_learning_rate * learning_rate_scale

        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=new_lr)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=new_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        return optimizer

    def copy(self) -> 'ModelWrapper':
        """Create a deep copy of the current ModelWrapper instance."""
        # Create a new instance with the same configuration
        new_instance = ModelWrapper(
            model_name=self.model_name,
            model_type=self.model_type,
            model_class=self.model_class,
            processor_class=self.processor_class,
            is_huggingface=self.is_huggingface,
            input_shape=self.input_shape,
            debug=self.debug.copy(),
            **copy.deepcopy(self.model_kwargs)
        )
        
        # Create the base model
        new_instance.create_base_model()
        
        # Copy weights if the original model has been trained
        if self.model is not None:
            new_instance.model.set_weights(self.model.get_weights())
        
        return new_instance

    def _validate_activation_replacement(self):
        """Validate that necessary activation replacement parameters are set."""
        if self.activation_function_to_replace is None:
            raise ValueError("activation_type_to_replace must be set in the config before performing activation replacement operations.")
        if self.activation_function is None:
            raise ValueError("activation_function must be set before performing activation replacement operations.")

    def generate_analysis_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        history: Union[tf.keras.callbacks.History, Dict],
        report_name: str = "analysis_report.pdf"
    ) -> None:
        """
        Generate a PDF report analyzing the model's performance.

        Args:
            X_test: Test data.
            y_test: True labels for the test data.
            history: Training history.
            report_name: Filename for the generated report.
        """
        with PdfPages(report_name) as pdf:
            # Page 1: Training Metrics
            if history is not None:
                self._plot_training_history(history, pdf)

            # Page 2: Confusion Matrix
            self._plot_confusion_matrix(X_test, y_test, pdf)

            # Page 3: Classification Report
            self._plot_classification_report(X_test, y_test, pdf)

            # Additional pages can be added as needed...
            # For more complex cases, stub methods can be provided for future implementation.

        print(f"Analysis report saved to {report_name}")

    def _plot_training_history(self, history, pdf):
        """
        Plot training and validation accuracy and loss over epochs.

        Args:
            history: Training history.
            pdf: PdfPages object to save the plot.
        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 10))

        # Check if history is a History object or a dictionary
        if isinstance(history, tf.keras.callbacks.History):
            hist = history.history
        elif isinstance(history, dict):
            hist = history
        else:
            print("Unsupported history format.")
            return

        # Plot accuracy
        if 'accuracy' in hist:
            axs[0].plot(hist['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in hist:
                axs[0].plot(hist['val_accuracy'], label='Validation Accuracy')
            axs[0].set_title('Model Accuracy')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Accuracy')
            axs[0].legend()
            axs[0].grid(True)
        else:
            axs[0].text(0.5, 0.5, 'Accuracy not available.', horizontalalignment='center', verticalalignment='center')

        # Plot loss
        if 'loss' in hist:
            axs[1].plot(hist['loss'], label='Training Loss')
            if 'val_loss' in hist:
                axs[1].plot(hist['val_loss'], label='Validation Loss')
            axs[1].set_title('Model Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Loss')
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].text(0.5, 0.5, 'Loss not available.', horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    def _plot_confusion_matrix(self, X_test, y_test, pdf):
        """
        Plot the confusion matrix.

        Args:
            X_test: Test data.
            y_test: True labels.
            pdf: PdfPages object to save the plot.
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        # Set tick marks
        classes = np.unique(y_true)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        # Annotate each cell
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{}'.format(z), ha='center', va='center')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    def _plot_classification_report(self, X_test, y_test, pdf):
        """
        Generate and plot the classification report.

        Args:
            X_test: Test data.
            y_test: True labels.
            pdf: PdfPages object to save the plot.
        """
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report = classification_report(y_true, y_pred_classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        fig, ax = plt.subplots(figsize=(8.5, len(report_df) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=np.round(report_df.values, 2),
                         rowLabels=report_df.index,
                         colLabels=report_df.columns,
                         cellLoc='center',
                         loc='center')
        table.scale(1, 1.5)
        ax.set_title('Classification Report')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    def generate_comparison_report(
        original_model: 'ModelWrapper',
        modified_model: 'ModelWrapper',
        X_test: np.ndarray,
        y_test: np.ndarray,
        original_history: Union[tf.keras.callbacks.History, Dict],
        modified_history: Union[tf.keras.callbacks.History, Dict],
        report_name: str = "comparison_report.pdf"
    ) -> None:
        """
        Generate a PDF report comparing the original and modified models.

        Args:
            original_model: The original ModelWrapper instance.
            modified_model: The modified ModelWrapper instance.
            X_test: Test data.
            y_test: True labels for the test data.
            original_history: Training history of the original model.
            modified_history: Training history of the modified model.
            report_name: Filename for the generated report.
        """
        with PdfPages(report_name) as pdf:
            # Page 1: Training Metrics Comparison
            ModelWrapper._plot_training_histories(
                original_history, modified_history, pdf
            )

            # Page 2: Confusion Matrices
            ModelWrapper._plot_confusion_matrices(
                original_model, modified_model, X_test, y_test, pdf
            )

            # Page 3: Classification Reports
            ModelWrapper._plot_classification_reports(
                original_model, modified_model, X_test, y_test, pdf
            )

            # Page 4: Prediction Distributions
            ModelWrapper._plot_prediction_distributions(
                original_model, modified_model, X_test, pdf
            )

            # Page 5: Difference in Predictions
            ModelWrapper._plot_prediction_differences(
                original_model, modified_model, X_test, pdf
            )

            # Additional pages can be added as needed...

        print(f"Comparison report saved to {report_name}")

    @staticmethod
    def _plot_training_histories(
        original_history, modified_history, pdf
    ):
        """
        Plot training and validation accuracy and loss over epochs for both models.

        Args:
            original_history: Training history of the original model.
            modified_history: Training history of the modified model.
            pdf: PdfPages object to save the plot.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Extract histories
        orig_hist = (
            original_history.history if isinstance(original_history, tf.keras.callbacks.History) 
            else original_history
        )
        mod_hist = (
            modified_history.history if isinstance(modified_history, tf.keras.callbacks.History) 
            else modified_history
        )

        # Plot loss
        axs[0, 0].plot(orig_hist.get('loss', []), label='Training Loss')
        axs[0, 0].plot(orig_hist.get('val_loss', []), label='Validation Loss')
        axs[0, 0].set_title('Original Model Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(mod_hist.get('loss', []), label='Training Loss')
        axs[0, 1].plot(mod_hist.get('val_loss', []), label='Validation Loss')
        axs[0, 1].set_title('Modified Model Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot accuracy
        axs[1, 0].plot(orig_hist.get('accuracy', []), label='Training Accuracy')
        axs[1, 0].plot(orig_hist.get('val_accuracy', []), label='Validation Accuracy')
        axs[1, 0].set_title('Original Model Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(mod_hist.get('accuracy', []), label='Training Accuracy')
        axs[1, 1].plot(mod_hist.get('val_accuracy', []), label='Validation Accuracy')
        axs[1, 1].set_title('Modified Model Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    @staticmethod
    def _plot_confusion_matrices(
        original_model, modified_model, X_test, y_test, pdf
    ):
        """
        Plot the confusion matrices for both models.

        Args:
            original_model: The original ModelWrapper instance.
            modified_model: The modified ModelWrapper instance.
            X_test: Test data.
            y_test: True labels.
            pdf: PdfPages object to save the plot.
        """
        y_pred_orig = original_model.model.predict(X_test)
        y_pred_mod = modified_model.model.predict(X_test)
        y_pred_classes_orig = np.argmax(y_pred_orig, axis=1)
        y_pred_classes_mod = np.argmax(y_pred_mod, axis=1)
        y_true = np.argmax(y_test, axis=1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Original Model Confusion Matrix
        cm_orig = confusion_matrix(y_true, y_pred_classes_orig)
        axs[0].matshow(cm_orig, cmap=plt.cm.Blues)
        axs[0].set_title('Original Model Confusion Matrix')
        axs[0].set_xlabel('Predicted')
        axs[0].set_ylabel('True')

        # Modified Model Confusion Matrix
        cm_mod = confusion_matrix(y_true, y_pred_classes_mod)
        axs[1].matshow(cm_mod, cmap=plt.cm.Blues)
        axs[1].set_title('Modified Model Confusion Matrix')
        axs[1].set_xlabel('Predicted')
        axs[1].set_ylabel('True')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    @staticmethod
    def _plot_classification_reports(
        original_model, modified_model, X_test, y_test, pdf
    ):
        """
        Generate and plot the classification reports for both models.

        Args:
            original_model: The original ModelWrapper instance.
            modified_model: The modified ModelWrapper instance.
            X_test: Test data.
            y_test: True labels.
            pdf: PdfPages object to save the plot.
        """
        y_pred_orig = original_model.model.predict(X_test)
        y_pred_mod = modified_model.model.predict(X_test)
        y_pred_classes_orig = np.argmax(y_pred_orig, axis=1)
        y_pred_classes_mod = np.argmax(y_pred_mod, axis=1)
        y_true = np.argmax(y_test, axis=1)

        report_orig = classification_report(y_true, y_pred_classes_orig, output_dict=True)
        report_mod = classification_report(y_true, y_pred_classes_mod, output_dict=True)

        report_df_orig = pd.DataFrame(report_orig).transpose()
        report_df_mod = pd.DataFrame(report_mod).transpose()

        fig, axs = plt.subplots(1, 2, figsize=(17, len(report_df_orig) * 0.5))
        
        # Original Model Classification Report
        axs[0].axis('tight')
        axs[0].axis('off')
        table_orig = axs[0].table(
            cellText=np.round(report_df_orig.values, 2),
            rowLabels=report_df_orig.index,
            colLabels=report_df_orig.columns,
            cellLoc='center',
            loc='center'
        )
        table_orig.scale(1, 1.5)
        axs[0].set_title('Original Model Classification Report')

        # Modified Model Classification Report
        axs[1].axis('tight')
        axs[1].axis('off')
        table_mod = axs[1].table(
            cellText=np.round(report_df_mod.values, 2),
            rowLabels=report_df_mod.index,
            colLabels=report_df_mod.columns,
            cellLoc='center',
            loc='center'
        )
        table_mod.scale(1, 1.5)
        axs[1].set_title('Modified Model Classification Report')

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    @staticmethod
    def _plot_prediction_distributions(
        original_model, modified_model, X_test, pdf
    ):
        """
        Plot the distributions of the model outputs (probabilities or logits) for both models.

        Args:
            original_model: The original ModelWrapper instance.
            modified_model: The modified ModelWrapper instance.
            X_test: Test data.
            pdf: PdfPages object to save the plot.
        """
        y_pred_orig = original_model.model.predict(X_test)
        y_pred_mod = modified_model.model.predict(X_test)

        # Flatten the predictions to 1D arrays
        y_pred_orig_flat = y_pred_orig.flatten()
        y_pred_mod_flat = y_pred_mod.flatten()

        plt.figure(figsize=(12, 6))
        plt.hist(y_pred_orig_flat, bins=50, alpha=0.5, label='Original Model')
        plt.hist(y_pred_mod_flat, bins=50, alpha=0.5, label='Modified Model')
        plt.title('Prediction Distributions')
        plt.xlabel('Predicted Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    @staticmethod
    def _plot_prediction_differences(
        original_model, modified_model, X_test, pdf
    ):
        """
        Plot the differences between the predictions of the original and modified models.

        Args:
            original_model: The original ModelWrapper instance.
            modified_model: The modified ModelWrapper instance.
            X_test: Test data.
            pdf: PdfPages object to save the plot.
        """
        y_pred_orig = original_model.model.predict(X_test)
        y_pred_mod = modified_model.model.predict(X_test)
        differences = y_pred_orig - y_pred_mod

        # Plot histogram of differences
        plt.figure(figsize=(12, 6))
        plt.hist(differences.flatten(), bins=50, alpha=0.7)
        plt.title('Differences in Predictions (Original - Modified)')
        plt.xlabel('Difference')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Scatter plot of predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred_orig.flatten(), y_pred_mod.flatten(), alpha=0.5)
        plt.plot([y_pred_orig.min(), y_pred_orig.max()], [y_pred_orig.min(), y_pred_orig.max()], 'r--')
        plt.title('Original vs. Modified Predictions')
        plt.xlabel('Original Predictions')
        plt.ylabel('Modified Predictions')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
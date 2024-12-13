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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from PredictionDecoder import PredictionDecoder
from PassReport import SingleModelReport, ModelComparisonReport

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

        # Initialize compilation settings based on model type
        if model_type == "classification":
            self.base_settings = {
                'loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                'metrics': [
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
                ],
                'jit_compile': False
            }
        else:
            # Default settings for other model types
            self.base_settings = {
                'loss': 'mse',
                'metrics': ['mae'],
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

        # Initialize callbacks as an empty list
        self.callbacks: List[tf.keras.callbacks.Callback] = []

        # If you have callbacks defined in the config, initialize them here
        if config_path:
            # Example: Load callbacks from a config file
            config = self.load_config(config_path)
            self.callbacks = self.initialize_callbacks(config.get('callbacks', []))
        
        # Initialize other attributes as needed
        self.activation_layer_pairs: List[Tuple[tf.keras.layers.Layer, tf.keras.layers.Activation]] = []

        # Call this method during initialization
        self._cache_model_config()

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

        if isinstance(self.model, tf.keras.Sequential):
            new_layers = []
            
            for i, layer in enumerate(self.model.layers):
                config = layer.get_config()
                is_final_layer = i == len(self.model.layers) - 1
                
                # Check if the layer has 'activation' in its config
                has_activation_in_config = (
                    'activation' in config and
                    config['activation'] is not None and
                    config['activation'] != 'linear'
                )

                # Only replace non-softmax activations and non-final layers
                if (has_activation_in_config and 
                    config['activation'] == self.config['activation_type_to_replace'] and
                    not (is_final_layer and config['activation'] == 'softmax')):
                    
                    # Remove activation from config
                    original_activation = config['activation']
                    config['activation'] = 'linear'

                    if self.debug.get('print_activation_replacement_debug', False):
                        print(f"Splitting activation in layer {layer.name}")

                    # Clone layer without activation
                    new_layer = layer.__class__.from_config(config)

                    # Build and set weights if the layer has weights
                    if layer.get_weights() and hasattr(layer, 'input_shape'):
                        new_layer.build(layer.input_shape)
                        new_layer.set_weights(layer.get_weights())

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
            
            # Recompile if original was compiled
            if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                self.model = new_model
                self.compile_model()
            else:
                self.model = new_model
            
            return self.model

        else:
            # Handle Functional API models
            inputs = self.model.input
            layer_outputs = {}
            new_layers = {}
            
            # First pass: create new layers without activation
            for layer in self.model.layers:
                config = layer.get_config()
                is_output_layer = layer.name == self.model.output.name.split('/')[0]
                
                # Check if the layer has 'activation' in its config
                has_activation_in_config = (
                    'activation' in config and
                    config['activation'] is not None and
                    config['activation'] != 'linear'
                )

                # Only replace non-softmax activations and non-output layers
                if (has_activation_in_config and 
                    config['activation'] == self.config['activation_type_to_replace'] and
                    not (is_output_layer and config['activation'] == 'softmax')):
                    
                    # Remove activation from config
                    original_activation = config['activation']
                    config['activation'] = 'linear'

                    if self.debug.get('print_activation_replacement_debug', False):
                        print(f"Splitting activation in layer {layer.name}")

                    # Clone layer without activation
                    new_layer = layer.__class__.from_config(config)
                    
                    # Create new activation layer
                    activation_layer = tf.keras.layers.Activation(
                        activation=original_activation,
                        name=f"{layer.name}_activation"
                    )
                    
                    # Store both layers
                    new_layers[layer.name] = (new_layer, activation_layer)
                    
                    # Save the pair for replacement during retraining
                    self.activation_layer_pairs.append((new_layer, activation_layer))
                else:
                    # Keep layer as-is
                    new_layers[layer.name] = (layer, None)

            # Second pass: reconstruct model connections
            for layer in self.model.layers:
                current_layer, activation_layer = new_layers[layer.name]
                
                # Get input tensors for current layer
                if layer.name == self.model.layers[0].name:
                    # Input layer
                    layer_outputs[layer.name] = current_layer(inputs)
                else:
                    # Get inbound nodes
                    inbound_layers = layer._inbound_nodes[0].inbound_layers
                    if not isinstance(inbound_layers, list):
                        inbound_layers = [inbound_layers]
                    
                    # Get input tensors
                    input_tensors = [layer_outputs[l.name] for l in inbound_layers]
                    if len(input_tensors) == 1:
                        input_tensors = input_tensors[0]
                    
                    # Apply current layer
                    layer_outputs[layer.name] = current_layer(input_tensors)
                
                # Apply activation layer if it exists
                if activation_layer is not None:
                    layer_outputs[layer.name] = activation_layer(layer_outputs[layer.name])

            # Create new model
            outputs = layer_outputs[self.model.layers[-1].name]
            new_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.model.name)
            
            # Recompile if original was compiled
            if hasattr(self.model, 'optimizer'):
                self.model = new_model
                self.compile_model()
            else:
                self.model = new_model

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
        from tensorflow.keras.callbacks import History, EarlyStopping
        
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
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Determine number of classes from labels shape
        num_classes = y_train.shape[-1] if len(y_train.shape) > 1 else len(np.unique(y_train))
        
        # Update compilation settings based on number of classes
        if num_classes > 2:
            self.base_settings.update({
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy', 'categorical_accuracy']
            })
        else:
            self.base_settings.update({
                'loss': 'binary_crossentropy',
                'metrics': ['accuracy', 'binary_accuracy']
            })
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        
        history = {}
        
        if retraining_type == 'all':
            history = self._retrain_all(train_dataset, val_dataset, epochs, callbacks=[early_stopping])
        elif retraining_type == 'iterative':
            history = self._retrain_iterative(train_dataset, val_dataset, epochs, callbacks=[early_stopping])
        elif retraining_type == 'batched':
            history = self._retrain_batched(train_dataset, val_dataset, activation_replacement_batch_size, epochs, callbacks=[early_stopping])
        
        return history

    def _retrain_all(self, train_dataset, val_dataset, epochs, callbacks=None):
        """Retrain all layers at once."""
        history = {}
        config = self.config['retraining']
        
        callbacks = callbacks or []
        
        # Optional: Train before activation replacement
        if config.get('train_before_activation', False):
            print("Training before activation replacement...")
            self.model.compile(
                optimizer=self.get_optimizer(),
                loss=self.base_settings['loss'],
                metrics=self.base_settings['metrics']
            )
            hist_before = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks
            )
            history['train_before_activation'] = hist_before.history
        
        # Replace all specified activations
        print("Replacing all specified activation functions...")
        for activation_layer in self.get_activation_layers():
            self.replace_activation(activation_layer, self.activation_function)
        
        # Optional: Train after activation replacement
        if config.get('train_after_activation', False):
            print("Training after activation replacement...")
            self.reset_optimizer(learning_rate_scale=0.1)
            self.model.compile(
                optimizer=self.get_optimizer(),
                loss=self.base_settings['loss'],
                metrics=self.base_settings['metrics']
            )
            hist_after = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks
            )
            history['train_after_activation'] = hist_after.history
        
        return history

    def _retrain_iterative(self, train_dataset, val_dataset, epochs, callbacks=None):
        """Iteratively retrain layers."""
        history = {'iterative': []}
        activation_layers = self.get_activation_layers()
        callbacks = callbacks or []

        # Save initial weights
        initial_weights = [layer.get_weights() for layer in self.model.layers]

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

            trainable_layers = []
            if prev_layer:
                prev_layer.trainable = True
                trainable_layers.append(prev_layer.name)
                if self.debug.get('print_retrain_debug', False):
                    print(f"Training layer before activation: {prev_layer.name}")

            if next_layer:
                next_layer.trainable = True
                trainable_layers.append(next_layer.name)
                if self.debug.get('print_retrain_debug', False):
                    print(f"Training layer after activation: {next_layer.name}")

            # Create optimizer with scaled learning rate
            optimizer = self.get_optimizer(learning_rate_scale=0.1)

            # Compile the model
            self.model.compile(
                optimizer=optimizer,
                loss=self.base_settings['loss'],
                metrics=self.base_settings['metrics']
            )

            # Train the model
            print(f"Retraining after replacing activation in layer {activation_layer.name}...")
            hist = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks
            )

            # Check if training improved the model
            val_loss = hist.history['val_loss'][-1]
            if val_loss > hist.history['val_loss'][0] * 1.1:  # 10% worse
                print("Training degraded model performance. Reverting weights...")
                # Revert weights for trainable layers
                for layer, weights in zip(self.model.layers, initial_weights):
                    if layer.name in trainable_layers:
                        layer.set_weights(weights)

            # Record history
            history_entry = {
                'layer': idx,
                'layer_name': activation_layer.name,
                'trainable_layers': trainable_layers,
                'phase': 'post_activation_replacement',
                'history': hist.history,
                'reverted': val_loss > hist.history['val_loss'][0] * 1.1
            }
            history['iterative'].append(history_entry)

        return history

    def _retrain_batched(self, train_dataset, val_dataset, activation_replacement_batch_size, epochs, callbacks=None):
        """Batched retraining based on activation_replacement_batch_size."""
        history = {'batched': []}
        activation_layers = self.get_activation_layers()
        num_batches = (len(activation_layers) + activation_replacement_batch_size - 1) // activation_replacement_batch_size
        callbacks = callbacks or []

        # Save initial weights
        initial_weights = [layer.get_weights() for layer in self.model.layers]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * activation_replacement_batch_size
            end_idx = min((batch_idx + 1) * activation_replacement_batch_size, len(activation_layers))
            batch_layers = activation_layers[start_idx:end_idx]

            if self.debug.get('print_retrain_debug', False):
                print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

            trainable_layers = []
            
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

                if prev_layer and prev_layer.name not in trainable_layers:
                    prev_layer.trainable = True
                    trainable_layers.append(prev_layer.name)
                    if self.debug.get('print_retrain_debug', False):
                        print(f"Training layer before activation: {prev_layer.name}")

                if next_layer and next_layer.name not in trainable_layers:
                    next_layer.trainable = True
                    trainable_layers.append(next_layer.name)
                    if self.debug.get('print_retrain_debug', False):
                        print(f"Training layer after activation: {next_layer.name}")

            # Create optimizer with scaled learning rate
            optimizer = self.get_optimizer(learning_rate_scale=0.1)

            # Compile the model
            self.model.compile(
                optimizer=optimizer,
                loss=self.base_settings['loss'],
                metrics=self.base_settings['metrics']
            )

            # Train the model
            print(f"Retraining after replacing activations in batch {batch_idx + 1}...")
            hist = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks
            )

            # Check if training improved the model
            val_loss = hist.history['val_loss'][-1]
            if val_loss > hist.history['val_loss'][0] * 1.1:  # 10% worse
                print("Training degraded model performance. Reverting weights...")
                # Revert weights for trainable layers
                for layer, weights in zip(self.model.layers, initial_weights):
                    if layer.name in trainable_layers:
                        layer.set_weights(weights)

            # Record history
            history_entry = {
                'batch': batch_idx,
                'layers': [layer.name for layer in batch_layers],
                'trainable_layers': trainable_layers,
                'history': hist.history,
                'reverted': val_loss > hist.history['val_loss'][0] * 1.1
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
        """Create a deep copy of the model wrapper, including a reinitialized optimizer."""
        new_wrapper = copy.deepcopy(self)
        
        # Get the original model's configuration and weights
        config = self.model.get_config()
        weights = self.model.get_weights()
        
        # Create a new model with the same architecture
        new_wrapper.model = tf.keras.Sequential.from_config(config)
        new_wrapper.model.set_weights(weights)
        
        # Reinitialize the optimizer from the original optimizer's configuration
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_config = self.model.optimizer.get_config()
            optimizer_class = type(self.model.optimizer)
            new_optimizer = optimizer_class.from_config(optimizer_config)
            
            # Compile the new model with the reinitialized optimizer
            new_wrapper.model.compile(
                optimizer=new_optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics
            )
        else:
            # Compile with default settings if no optimizer exists
            new_wrapper.compile_model()
        
        return new_wrapper

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
        report = SingleModelReport(
            model=self,
            X_test=X_test,
            y_test=y_test,
            history=history,
            output_path=report_name
        )
        report.save()
        print(f"Analysis report saved to {report_name}")

    @staticmethod
    def generate_comparison_report(
        original_model: 'ModelWrapper',
        modified_model: 'ModelWrapper',
        X_test: np.ndarray,
        y_test: np.ndarray,
        original_history: Union[tf.keras.callbacks.History, Dict],
        modified_history: Union[tf.keras.callbacks.History, Dict],
        report_name: str = "comparison_report.pdf",
        max_points: int = 1000
    ) -> None:
        """
        Generate a PDF report comparing the original and modified models.
        """
        report = ModelComparisonReport(
            original_model=original_model,
            modified_model=modified_model,
            X_test=X_test,
            y_test=y_test,
            original_history=original_history,
            modified_history=modified_history,
            output_path=report_name
        )
        report.save()
        print(f"Comparison report saved to {report_name}")

    def validate_model_output(self, expected_classes: int = None):
        """
        Validate the model's output shape matches expected number of classes.
        
        Args:
            expected_classes: Expected number of output classes. If None, inferred from model.
        """
        output_shape = self.model.output_shape
        if len(output_shape) != 2:  # (batch_size, num_classes)
            raise ValueError(f"Expected 2D output shape (batch_size, num_classes), got {output_shape}")
        
        actual_classes = output_shape[-1]
        if expected_classes and actual_classes != expected_classes:
            raise ValueError(f"Model output shape mismatch. Expected {expected_classes} classes, got {actual_classes}")
        
        # Verify final layer activation
        final_layer = self.model.layers[-1]
        if not isinstance(final_layer, tf.keras.layers.Dense):
            raise ValueError(f"Expected final layer to be Dense, got {type(final_layer)}")
        
        final_activation = final_layer.activation
        if final_activation.__name__ != 'softmax':
            raise ValueError(f"Expected final activation to be softmax, got {final_activation.__name__}")

    def configure_for_classification(self, num_classes: int):
        """
        Configure the model wrapper for classification with specified number of classes.
        
        Args:
            num_classes: Number of output classes
        """
        self.base_settings.update({
            'loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            'metrics': [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=min(5, num_classes), name='top_k_accuracy')
            ]
        })
        
        # Add classification-specific callbacks
        self.callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6
            )
        ])

    def compile_model(self, override_settings=None):
        """Compile the model with stored or override settings."""
        settings = self.base_settings.copy()
        if override_settings:
            settings.update(override_settings)
        
        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            optimizer_config = self.model.optimizer.get_config()
            optimizer_class = type(self.model.optimizer)
            new_optimizer = optimizer_class.from_config(optimizer_config)
            
            self.model.compile(
                optimizer=new_optimizer,
                loss=settings.get('loss', self.model.loss),
                metrics=settings.get('metrics', self.model.metrics),
                **{k: v for k, v in settings.items() 
                   if k not in ['optimizer', 'loss', 'metrics']}
            )
        else:
            self.model.compile(
                optimizer=settings.get('optimizer', 'adam'),
                loss=settings.get('loss', self.model.loss),
                metrics=settings.get('metrics', self.model.metrics),
                **{k: v for k, v in settings.items() 
                   if k not in ['optimizer', 'loss', 'metrics']}
            )

    def _cache_model_config(self):
        """Cache model configuration information."""
        if self.model is not None:
            self.has_single_input = hasattr(self.model, 'input')
            self.input_shape = self.model.input_shape if self.has_single_input else self.model.inputs[0].shape
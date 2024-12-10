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
        model_name: str = "microsoft/resnet-50", 
        model_type: str = "image", 
        debug: Dict[str, bool] = None, 
        model_class: Optional[Any] = None, 
        processor_class: Optional[Any] = None, 
        is_huggingface: bool = True,
        input_shape: Tuple[int, ...] = (224, 224, 3),
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
        self.activation_function_to_replace = self.config.get('activation_type_to_replace', None)
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
            raise FileNotFoundError(f"Config file not found at {config_path}")

    def set_activation_function(self, activation_function):
        """Set the activation function to use for replacement."""
        self.activation_function = activation_function
        return self
        
    def create_base_model(self) -> 'ModelWrapper':
        """Initialize the model and processor based on model type"""
        if self.is_huggingface:
            if self.model_class is not None and self.processor_class is not None:
                self.model = self.model_class.from_pretrained(self.model_name)
                self.processor = self.processor_class.from_pretrained(self.model_name)
            else:
                raise ValueError("Model class and processor class must be provided for HuggingFace models.")
        else:
            # Handle native TensorFlow models
            if self.model_class is not None:
                if callable(self.model_class) and not isinstance(self.model_class, type):
                    # Handle custom model function
                    print("Creating custom model with input_shape and model_kwargs...")
                    print(f"model_kwargs: {self.model_kwargs}")  # Debugging
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

    def retrain(self, train_data: np.ndarray, train_labels: np.ndarray, retrain_type: RetrainType):
        """
        Retrain the model using the specified strategy.

        Args:
            train_data: Training data.
            train_labels: Training labels.
            retrain_type: RetrainType enum value.

        Returns:
            A dictionary containing training histories.
        """
        from tensorflow.keras.callbacks import History

        # Prepare datasets
        batch_size = self.config['training']['batch_size']
        validation_split = self.config['training']['validation_split']

        # Split data
        split_index = int(len(train_data) * (1 - validation_split))
        x_train, x_val = train_data[:split_index], train_data[split_index:]
        y_train, y_val = train_labels[:split_index], train_labels[split_index:]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

        history = {}

        if retrain_type == RetrainType.ALL:
            history = self._retrain_all(train_dataset, val_dataset)
        elif retrain_type == RetrainType.ITERATIVE:
            history = self._retrain_iterative(train_dataset, val_dataset)
        elif retrain_type == RetrainType.BATCHED:
            history = self._retrain_batched(train_dataset, val_dataset)
        else:
            raise ValueError(f"Unknown retrain_type: {retrain_type}")

        return history

    def _retrain_all(self, train_dataset, val_dataset):
        """Retrain all layers at once."""
        history = {}
        config = self.config['retraining']['all_mode']
        total_epochs = 0

        if config['train_before_activation']:
            epochs = config['epochs']
            optimizer = self.get_optimizer(learning_rate_scale=0.1)
            self.model.compile(
                optimizer=optimizer,
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )
            hist_before = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )
            history['train_before_activation'] = hist_before.history
            total_epochs += epochs

        # Replace activations
        self.replace_all_activations(self.activation_function_to_replace, self.activation_function)

        if config['train_after_activation']:
            epochs = config['epochs']
            optimizer = self.get_optimizer(learning_rate_scale=0.1)
            self.model.compile(
                optimizer=optimizer,
                loss=self.config['training']['loss'],
                metrics=self.config['training']['metrics']
            )
            hist_after = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )
            history['train_after_activation'] = hist_after.history
            total_epochs += epochs

        return history

    def _retrain_iterative(self, train_dataset, val_dataset):
        """Iteratively retrain layers."""
        history = {'iterative': []}
        config = self.config['retraining']['iterative_mode']
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

            # Retrain
            epochs = config['epochs']
            hist = self.model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset
            )

            # Collect history
            step_history = {
                'layer': idx,
                'phase': 'iteration',
                'history': hist.history
            }
            history['iterative'].append(step_history)

        return history

    def _retrain_batched(self, train_dataset, val_dataset):
        """Retrain layers in batches."""
        history = {'batched': []}
        config = self.config['retraining']['batched_mode']
        batch_size = config['batch_size']
        activation_layers = self.get_activation_layers()

        num_batches = (len(activation_layers) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_history = {}
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(activation_layers))
            batch_layers = activation_layers[start_idx:end_idx]

            if config['train_before_activation']:
                epochs = config['epochs']
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                hist_before = self.model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset
                )
                batch_history['batch'] = batch_idx
                batch_history['phase'] = 'before'
                batch_history['history'] = hist_before.history
                history['batched'].append(batch_history)

            # Replace activations in batch
            for activation_layer in batch_layers:
                self.replace_activation(activation_layer, self.activation_function)

            if config['train_after_activation']:
                epochs = config['epochs']
                self.reset_optimizer(learning_rate_scale=0.1)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                hist_after = self.model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset
                )
                batch_history = {}
                batch_history['batch'] = batch_idx
                batch_history['phase'] = 'after'
                batch_history['history'] = hist_after.history
                history['batched'].append(batch_history)

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
        if self.activation_function is None:
            raise ValueError("activation_function must be set before performing activation replacement operations")
        if self.activation_function_to_replace is None:
            raise ValueError("activation_function_to_replace must be set (either through config or directly) before performing activation replacement operations")
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
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        config_path: Optional[str] = None
    ) -> None:
        # Enable eager execution at the start
        tf.config.run_functions_eagerly(True)
        
        # Initialize model-specific attributes
        self.model_name = model_name
        self.model_type = model_type
        self.debug = debug
        self.model = None
        self.processor = None
        self.activation_function = None
        self.model_class = model_class
        self.processor_class = processor_class
        self.is_huggingface = is_huggingface
        self.input_shape = input_shape
        self.config = self._load_config(config_path)
        self.activation_layer_pairs = []  # List of (original_layer, activation_layer) tuples

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

    def set_activation_function(self, activation_function: Callable) -> 'ModelWrapper':
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
                self.model = self.model_class(
                    weights='imagenet',
                    include_top=True,
                    input_shape=self.input_shape
                )
                # For TF models, processor is typically just a preprocessing function
                self.processor = self.processor_class
            else:
                raise ValueError("Model class must be provided for TensorFlow models.")
        return self

    def split_activation_layers(self) -> tf.keras.Model:
        """Split activation layers from their parent layers"""
        # Get the base model depending on the type
        if hasattr(self.model, 'keras_model'):
            # HuggingFace models with direct Keras model
            base_model = self.model.keras_model
        elif hasattr(self.model, 'vit') and hasattr(self.model.vit, 'encoder'):
            # HuggingFace ViT models
            base_model = self.model.vit
            image_size = self.model.config.image_size
            num_channels = 3
            inputs = tf.keras.Input(shape=(image_size, image_size, num_channels))
            x = inputs
            x = base_model(x)
            base_model = tf.keras.Model(inputs=inputs, outputs=x)
        elif isinstance(self.model, tf.keras.Model):
            # Native TensorFlow/Keras models
            base_model = self.model
        else:
            raise ValueError("This model type is not currently supported for activation splitting")
        
        inputs = base_model.input
        if inputs is None:
            raise ValueError("The model has no inputs defined. Ensure the model is properly initialized.")
        
        x = inputs
        new_layers = []

        # Create a dictionary to store the tensor outputs for each layer
        layer_outputs = {}
        layer_outputs[base_model.input.name] = x
        
        for layer in base_model.layers:
            if self.debug['print_network_split_debug']:
                print(f"Processing layer: {layer.name}, type: {type(layer)}")
            
            # Skip input layer
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
                
            # Get the correct input for this layer
            if isinstance(layer, (tf.keras.layers.Add, tf.keras.layers.Concatenate)):
                layer_input = [layer_outputs[inp.name] for inp in layer.input]
            else:
                layer_input = layer_outputs[layer.input.name] if hasattr(layer, 'input') and layer.input is not None else x
                
            # Special handling for BatchNormalization
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                if self.debug['print_network_split_debug']:
                    print(f"Found BatchNorm layer: {layer.name}")
                x = layer(layer_input)  # Use layer_input instead of x
                new_layers.append(layer)
                layer_outputs[layer.output.name] = x
                continue
                
            # Get layer config and check for activation
            config = layer.get_config()
            has_activation = ('activation' in config and 
                            config['activation'] is not None and 
                            config['activation'] != 'linear')
            
            if has_activation:
                # Track the layer pair if it's the type we want to replace
                if config['activation'] == self.config['activation_type_to_replace']:
                    # Here's the general idea:
                    #    1. Replace the layer's activation with a linear activation.
                    #    2. Append a new activation layer after this layer with our polynomial approximation for the activation function.
                    # AFAICT, this is equivalent to replacing the activation function with a polynomial approximation.
                    # Consider, for example a layer with ReLU activation:
                    #    output = relu(W*x+b)
                    # becomes
                    #    temp = linear(W*x+b) = W*x+b
                    #    output = poly(temp) = poly(W*x+b) = a0 + a1*(W*x+b) + a2*(W*x+b)^2 + ...
                    # which is equivalent to:
                    #    output = poly(W*x+b) = a0 + a1*(W*x+b) + a2*(W*x+b)^2 + ...

                    # Store original weights if the layer has them
                    original_weights = layer.get_weights() if hasattr(layer, 'get_weights') else None
                    
                    # Print debug info about the activation replacement
                    print(f"\nSplitting activation for layer: {layer.name}")
                    print(f"Original activation: {config['activation']}")
                    if original_weights:
                        print(f"Original weights shapes: {[w.shape for w in original_weights]}")
                    
                    # Special handling for final predictions layer
                    if layer.name == 'predictions':
                        # Keep the original layer with its weights
                        x = layer(layer_input)
                        new_layers.append(layer)
                        layer_outputs[layer.output.name] = x
                        continue
                    
                    # Remove activation from layer
                    config['activation'] = 'linear'
                    
                    # Clone layer without activation
                    if hasattr(layer, '_keras_api_names'):
                        layer_type = layer.__class__
                        new_layer = layer_type.from_config(config)
                    else:
                        new_layer = layer.__class__.from_config(config)
                    
                    # Restore original weights if they existed
                    if original_weights:
                        new_layer.set_weights(original_weights)
                        print(f"Restored weights shapes: {[w.shape for w in new_layer.get_weights()]}")
                    
                    # Add the layer and its activation separately
                    x = new_layer(layer_input)
                    new_layers.append(new_layer)
                    
                    # Add separate activation layer
                    activation_layer = tf.keras.layers.Activation(config['activation'], name=f'{layer.name}_activation')
                    x = activation_layer(x)
                    new_layers.append(activation_layer)
                    
                    self.activation_layer_pairs.append((new_layer, activation_layer))
                else:
                    # This is an activation we don't want to replace, add as-is
                    x = layer(layer_input)
                    new_layers.append(layer)
            else:
                # Layer doesn't have activation, add as-is
                x = layer(layer_input)
                new_layers.append(layer)
            
            # Store the output tensor
            layer_outputs[layer.output.name] = x
        
        # Create new model
        new_model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # Create a mapping of old to new layers
        old_to_new = {}
        new_idx = 0
        
        for old_layer in base_model.layers:
            if isinstance(old_layer, tf.keras.layers.InputLayer):
                continue
                
            if isinstance(old_layer, tf.keras.layers.BatchNormalization):
                if self.debug['print_network_split_debug']:
                    print(f"Skipping mapping for BatchNorm: {old_layer.name}")
                continue
                
            # Find the corresponding new layer
            while new_idx < len(new_layers):
                new_layer = new_layers[new_idx]
                if isinstance(new_layer, tf.keras.layers.BatchNormalization):
                    new_idx += 1
                    continue
                    
                # Map the layers if they have the same base name (ignoring activation suffix)
                old_base_name = old_layer.name.replace('_activation', '')
                new_base_name = new_layer.name.replace('_activation', '')
                
                if old_base_name == new_base_name:
                    old_to_new[old_layer] = new_layer
                    if self.debug['print_network_split_debug']:
                        print(f"Mapped {old_layer.name} -> {new_layer.name}")
                    new_idx += 1
                    break
                new_idx += 1
        
        # Copy weights using the mapping
        for old_layer, new_layer in old_to_new.items():
            if hasattr(old_layer, 'get_weights'):
                weights = old_layer.get_weights()
                if weights:  # Only set weights if layer has weights
                    if self.debug['print_network_split_debug']:
                        print(f"Copying weights from {old_layer.name} to {new_layer.name}")
                    try:
                        new_layer.set_weights(weights)
                    except ValueError as e:
                        if self.debug['print_network_split_debug']:
                            print(f"Failed to copy weights: {e}")
                        continue
        
        return new_model
        
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

    def replace_activation(self, layer, new_activation):
        """Replace activation function and add regularization"""
        if hasattr(layer, 'activation'):
            layer.activation = new_activation
            
            # Add L2 regularization to kernel if available
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

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
    
    def get_trainable_layer_before(self, activation_layer: tf.keras.layers.Layer) -> Optional[tf.keras.layers.Layer]:
        """Get the first trainable layer before the given activation layer"""
        layer = self.get_layer_before(activation_layer)
        # We can't check the trainable attribute, because we just set it to False.
        # We should check for weights instead.  
        while layer is not None and len(layer.weights) == 0:
            layer = self.get_layer_before(layer)
        return layer

    def get_trainable_layer_after(self, activation_layer: tf.keras.layers.Layer) -> Optional[tf.keras.layers.Layer]:
        """Get the first trainable layer after the given activation layer"""
        layer = self.get_layer_after(activation_layer)
        # We can't check the trainable attribute, because we just set it to False.
        # We should check for weights instead.  
        while layer is not None and len(layer.weights) == 0:
            layer = self.get_layer_after(layer)
        return layer


    def add_layer_before(self, layer: tf.keras.layers.Layer, new_layer: tf.keras.layers.Layer) -> 'ModelWrapper':
        """Add a layer before the specified layer"""
        self.model.layers.insert(self.model.layers.index(layer), new_layer)
        return self

    def batch_renormalize_before_activation(self, activation_layer: tf.keras.layers.Activation) -> 'ModelWrapper':
        """Batch normalize the layer before the specified activation layer"""
        layer_before = self.get_layer_before(activation_layer)
        layer_before.add_before(tf.keras.layers.BatchNormalization())
        return self

    def retrain(self, train_data: np.ndarray, train_labels: np.ndarray, retrain_type: RetrainType = RetrainType.ITERATIVE) -> 'ModelWrapper':
        """Retrain the model with the specified approach"""
        training_config = self.config['training']
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_data(train_data, train_labels)
        
        # Debug dataset structure
        print("\nDebugging dataset structure:")
        for x, y in train_dataset.take(1):
            print(f"Input shape: {x.shape}")
            print(f"Label shape: {y.shape}")
            print(f"Input dtype: {x.dtype}")
            print(f"Label dtype: {y.dtype}")
        
        # Configure optimizer
        optimizer_name = training_config['optimizer'].lower()
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_config['learning_rate']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Debug model structure
        print("\nDebugging model structure:")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        
        # Configure base model compilation settings
        self.base_compile_settings = {
            'optimizer': optimizer,
            'loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Changed from_logits to False
            'metrics': [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
            'jit_compile': False
        }
        
        # Initial compilation
        self.model.compile(**self.base_compile_settings)
        
        # Execute retraining strategy
        if retrain_type == RetrainType.ITERATIVE:
            self._retrain_iterative(train_dataset, val_dataset)
        elif retrain_type == RetrainType.ALL:
            self._retrain_all(train_dataset, val_dataset)
        elif retrain_type == RetrainType.BATCHED:
            self._retrain_batched(train_dataset, val_dataset)
        
        return self

    def _retrain_iterative(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> 'ModelWrapper':
        """Retrain the model iteratively, one activation layer at a time"""
        config = self.config['retraining']['iterative_mode']
        
        for idx, layer_pair in enumerate(self.activation_layer_pairs):
            print(f"\nProcessing activation layer {idx+1}/{len(self.activation_layer_pairs)}: {layer_pair[1].name}")
            
            # 1. Reset all layers to non-trainable
            self._reset_trainable_layers()
            
            # Get the activation layer from the pair
            activation_layer = layer_pair[1]  # The second element is the activation layer
            
            # 2. Get surrounding layers
            prev_layer = self.get_trainable_layer_before(activation_layer)
            next_layer = self.get_trainable_layer_after(activation_layer)
            
            # 3. Warmup phase - train only surrounding layers
            if prev_layer:
                prev_layer.trainable = True
                print(f"Training layer before {activation_layer.name}: {prev_layer.name}")
            if next_layer:
                next_layer.trainable = True
                print(f"Training layer after {activation_layer.name}: {next_layer.name}")
                
            # 4. Replace activation
            self.replace_activation(activation_layer, self.activation_function)
            
            # 5. Compile with very low learning rate for warmup
            self.reset_optimizer(learning_rate_scale=0.001)
            self.model.compile(**self.current_compile_settings)
            
            # 6. Warmup training
            self._fit(
                train_dataset,
                epochs=2,
                validation_data=val_dataset,
                callbacks=self._get_callbacks()
            )
            
            # 7. Main training phase - include more surrounding layers
            self._gradually_unfreeze_layers(activation_layer, num_layers=2)
            
            # 8. Compile with slightly higher learning rate
            self.reset_optimizer(learning_rate_scale=0.01)
            self.model.compile(**self.current_compile_settings)
            
            # 9. Main training
            self._fit(
                train_dataset,
                epochs=config['epochs'],
                validation_data=val_dataset,
                callbacks=self._get_callbacks()
            )
        
        return self

    def _gradually_unfreeze_layers(self, center_layer, num_layers=2):
        """Unfreeze layers gradually outward from the center layer"""
        layer_list = self.model.layers
        center_idx = layer_list.index(center_layer)
        
        start_idx = max(0, center_idx - num_layers)
        end_idx = min(len(layer_list), center_idx + num_layers + 1)
        
        for layer in layer_list[start_idx:end_idx]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                continue  # Keep BN layers frozen
            layer.trainable = True

    def _retrain_all(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> 'ModelWrapper':
        """Retrain the model on all activation layers at once"""
        config = self.config['retraining']['all_mode']
        activation_layers = self.split_activation_layers
        
        # 1. Reset all layers to non-trainable
        for layer in self.model.layers:
            layer.trainable = False
        
        # 2. Make specific layers trainable and replace activations
        for activation_layer in activation_layers:
            if config['train_before_activation']:
                layer_before = self.get_layer_before(activation_layer)
                layer_before.trainable = True
            elif config['train_after_activation']:
                layer_after = self.get_layer_after(activation_layer)
                layer_after.trainable = True
            
            self.replace_activation(activation_layer, self.activation_function)
        
        # Reset optimizer with reduced learning rate for fine-tuning
        self.reset_optimizer(learning_rate_scale=0.01)
        
        # Compile with fresh settings
        self.model.compile(**self.current_compile_settings)
        
        # 4. Train once with all replacements
        self._fit(
            train_dataset, 
            epochs=config['epochs'],
            validation_data=val_dataset
        )
        
        return self

    def _retrain_batched(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> 'ModelWrapper':
        """Retrain the model in batches of activation layers"""
        config = self.config['retraining']['batch_mode']
        activation_layers = self.split_activation_layers
        
        for i in range(0, len(activation_layers), config['activation_replacement_batch_size']):
            # Get current batch of activation layers
            batch_layers = activation_layers[i:i + config['activation_replacement_batch_size']]
            
            # 1. Reset all layers to non-trainable
            for layer in self.model.layers:
                layer.trainable = False
            
            # 2. Make specific layers trainable and replace activations for this batch
            for activation_layer in batch_layers:
                if config['train_before_activation']:
                    layer_before = self.get_layer_before(activation_layer)
                    layer_before.trainable = True
                elif config['train_after_activation']:
                    layer_after = self.get_layer_after(activation_layer)
                    layer_after.trainable = True
                
                self.replace_activation(activation_layer, self.activation_function)
            
            # Reset optimizer with reduced learning rate for fine-tuning
            self.reset_optimizer(learning_rate_scale=0.01)
            
            # Compile with fresh settings
            self.model.compile(**self.current_compile_settings)
            
            # 4. Train on current batch of replacements
            self._fit(
                train_dataset, 
                epochs=config['epochs'],
                validation_data=val_dataset
            )
        
        return self

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

    def reset_optimizer(self, learning_rate_scale: float = 1.0):
        """Reset optimizer with scaled learning rate and update compilation settings"""
        training_config = self.config['training']
        optimizer_name = training_config['optimizer'].lower()
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_config['learning_rate'] * learning_rate_scale,
                clipnorm=1.0
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Update current compilation settings with new optimizer and base settings
        self.current_compile_settings = {
            'optimizer': optimizer,
            **self.base_settings  # Spread the base settings we defined in __init__
        }

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
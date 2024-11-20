from typing import Any, Dict, List, Union, Optional, Callable
import numpy as np
import tensorflow as tf
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoImageProcessor,
    ResNetForImageClassification
)
from PIL import Image

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

class ModelWrapper:
    def __init__(
        self, 
        model_name: str = "microsoft/resnet-50", 
        model_type: str = "image", 
        debug: Dict[str, bool] = None, 
        model_class: Optional[Any] = None, 
        processor_class: Optional[Any] = None, 
        is_huggingface: bool = True
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.debug = debug
        self.model = None
        self.processor = None
        self.model_class = model_class
        self.processor_class = processor_class
        self.is_huggingface = is_huggingface
        self.initialize_debug()

    def initialize_debug(self) -> None:
        if self.debug is None:
            self.debug = {
                'print_network_split_debug': False,
            }
        
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
                    input_shape=(224, 224, 3)
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
                # Store the activation type
                activation_type = config['activation']
                # Remove activation from layer
                config['activation'] = 'linear'
                
                # Clone layer without activation
                if hasattr(layer, '_keras_api_names'):
                    layer_type = layer.__class__
                    new_layer = layer_type.from_config(config)
                else:
                    new_layer = layer.__class__.from_config(config)
                
                # Add the layer and its activation separately
                x = new_layer(layer_input)
                new_layers.append(new_layer)
                
                # Add separate activation layer
                activation_layer = tf.keras.layers.Activation(activation_type, name=f'{layer.name}_activation')
                x = activation_layer(x)
                new_layers.append(activation_layer)
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
        
    def replace_activations(self, activation_function: Callable) -> 'ModelWrapper':
        replacements_made = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                # Replace sigmoid activations
                if layer.activation == tf.keras.activations.sigmoid:  
                    layer.activation = activation_function
                    replacements_made += 1
                # Keep ReLU as is
                elif layer.activation == tf.keras.activations.relu:   
                    continue
                # Keep other activations as is
                else:
                    continue
        
        if replacements_made == 0:
            print("Warning: No activation functions were replaced!")
        else:
            print(f"Replaced {replacements_made} activation functions")
        
        return self

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
            'top_k': 5
        }
        
        # Add HuggingFace specific info if needed
        if self.is_huggingface:
            model_info['id2label'] = self.model.config.id2label
        
        # Use PredictionDecoder to handle the outputs
        return PredictionDecoder.decode(outputs, model_info)
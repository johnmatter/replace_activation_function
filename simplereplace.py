import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from scipy.interpolate import approximate_taylor_polynomial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from keras.engine.keras_tensor import KerasTensor
from datasets import load_dataset, config
from transformers import AutoImageProcessor
import matplotlib.pyplot as plt
import pyfiglet
import json 
import pdb
import os
# config
debug = {
    'print_activation_functions': False,
    'print_network_config': False,
    'print_layer_outputs': False,
    'print_layer_activations': False,
    'print_network_split_debug': False,
}
piecewise_activation = False
resolution = 224

# load model
input_t = keras.Input(shape=(resolution, resolution, 3))
base_model = ResNet50(weights='imagenet', include_top=True, input_tensor=input_t)

# define activation function
def my_activation(x, coefficients):
    # return a polynomial activation function using the coefficients supplied
    # The coefficients are in increasing order of degree.
    # The degree of the polynomial is len(coefficients) - 1.
    # For example if coefficients are [1,2,3,0,5] the activation
    # function is 1 + 2x + 3x^2 + 5x^4 and the degree of the polynomial is 4.
    return sum([c * x**i for i, c in enumerate(coefficients)])

# print initial model
if debug['print_network_config']:
    print(pyfiglet.figlet_format("initial"))  
    print(json.dumps(base_model.get_config(),indent=4)) 

# Create a callable activation function using a Taylor series
# For what it's worth, HEIR uses -0.004 * x^3 + 0.197 * x + 0.5
# See https://github.com/google/heir/blob/f9e39963b863490eaec81e53b4b9d180a4603874/lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.cpp#L95-L100
degree = 3
activation_function = activations.sigmoid
if not piecewise_activation:
    # Single polynomial approximation with smaller scale
    poly_activation = approximate_taylor_polynomial(activation_function, 0, degree=degree, scale=1.0)
    
    def custom_poly_activation(x):
        if debug['print_layer_activations']:
            tf.print("Activation input stats:", 
                    "min:", tf.reduce_min(x),
                    "max:", tf.reduce_max(x),
                    "mean:", tf.reduce_mean(x),
                    "has_nan:", tf.reduce_any(tf.math.is_nan(x)))
        
        # Use wider clipping range
        x = tf.clip_by_value(x, -5.0, 5.0)  # Changed from -2.0, 2.0
        
        result = sum([c * K.pow(x, i) for i, c in enumerate(poly_activation.coefficients)])
        
        # Remove output clipping to allow full range of values
        # result = tf.clip_by_value(result, 0.0, 1.0)  # Comment this out
        
        if debug['print_layer_activations']:
            tf.print("Polynomial coefficients:", poly_activation.coefficients)
            tf.print("Activation output stats:", 
                    "min:", tf.reduce_min(result),
                    "max:", tf.reduce_max(result),
                    "mean:", tf.reduce_mean(result),
                    "has_nan:", tf.reduce_any(tf.math.is_nan(result)))

        return result
else:
    # Piecewise polynomial approximation with smaller ranges and scales
    poly_neg = approximate_taylor_polynomial(activation_function, -1, degree=degree, scale=0.5)
    poly_zero = approximate_taylor_polynomial(activation_function, 0, degree=degree, scale=0.5)
    poly_pos = approximate_taylor_polynomial(activation_function, 1, degree=degree, scale=0.5)
    
    def custom_poly_activation(x):

        if debug['print_layer_activations']:
            tf.print("Activation input stats:", 
                        "min:", tf.reduce_min(x),
                        "max:", tf.reduce_max(x),
                        "mean:", tf.reduce_mean(x),
                        "has_nan:", tf.reduce_any(tf.math.is_nan(x)))
    
        # Clip values to prevent explosion
        x = tf.clip_by_value(x, -5.0, 5.0)
        
        # Similar debug for piecewise activation
        result = K.switch(x < -0.5,
                       sum([c * K.pow(x, i) for i, c in enumerate(poly_neg.coefficients)]),
                       K.switch(x > 0.5,
                               sum([c * K.pow(x, i) for i, c in enumerate(poly_pos.coefficients)]),
                               sum([c * K.pow(x, i) for i, c in enumerate(poly_zero.coefficients)])))
    
        if debug['print_layer_activations']:
            tf.print("Piecewise activation output stats:", 
                        "min:", tf.reduce_min(result),
                        "max:", tf.reduce_max(result),
                        "mean:", tf.reduce_mean(result),
                        "has_nan:", tf.reduce_any(tf.math.is_nan(result)))
        return result

# print activation function
if debug['print_activation_functions']:
    print(pyfiglet.figlet_format("activation"))
    if piecewise_activation:
        print("\nNegative region:\n", poly_neg)
        print("\nZero region:\n", poly_zero)
        print("\nPositive region:\n", poly_pos)
    else:
        print("\nSingle polynomial:\n", poly_activation)

def split_activation_layers(model):
    inputs = model.input
    x = inputs
    new_layers = []

    # Create a dictionary to store the tensor outputs for each layer
    layer_outputs = {}
    layer_outputs[model.input.name] = x
    
    for layer in model.layers:
        if debug['print_network_split_debug']:
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
            if debug['print_network_split_debug']:
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
            activation_layer = Activation(activation_type, name=f'{layer.name}_activation')
            x = activation_layer(x)
            new_layers.append(activation_layer)
        else:
            # Layer doesn't have activation, add as-is
            x = layer(layer_input)
            new_layers.append(layer)
        
        # Store the output tensor
        layer_outputs[layer.output.name] = x
    
    # Create new model
    new_model = Model(inputs=inputs, outputs=x)
    
    # Create a mapping of old to new layers
    old_to_new = {}
    new_idx = 0
    
    for old_layer in model.layers:
        if isinstance(old_layer, tf.keras.layers.InputLayer):
            continue
            
        if isinstance(old_layer, tf.keras.layers.BatchNormalization):
            if debug['print_network_split_debug']:
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
                if debug['print_network_split_debug']:
                    print(f"Mapped {old_layer.name} -> {new_layer.name}")
                new_idx += 1
                break
            new_idx += 1
    
    # Copy weights using the mapping
    for old_layer, new_layer in old_to_new.items():
        if hasattr(old_layer, 'get_weights'):
            weights = old_layer.get_weights()
            if weights:  # Only set weights if layer has weights
                if debug['print_network_split_debug']:
                    print(f"Copying weights from {old_layer.name} to {new_layer.name}")
                try:
                    new_layer.set_weights(weights)
                except ValueError as e:
                    if debug['print_network_split_debug']:
                        print(f"Failed to copy weights: {e}")
                    continue
    
    return new_model

# # Split the model before replacing activations
# base_model = split_activation_layers(base_model)

if debug['print_network_config']:
    print(pyfiglet.figlet_format("after splitting"))
    print(json.dumps(base_model.get_config(), indent=4))

# Now replace activation functions
for layer in base_model.layers:
    if isinstance(layer, Activation):
        # Replace sigmoid activations
        if layer.activation == activations.sigmoid:  
            layer.activation = custom_poly_activation
        # Keep ReLU as is
        elif layer.activation == activations.relu:   
            continue
        # Keep other activations as is. We can extend this to other activations later.
        else:
            continue

# compile model
base_model.compile(loss="categorical_crossentropy", optimizer='adam')

# print model after replacing activation function
if debug['print_network_config']: 
    print(pyfiglet.figlet_format("after replacing"))
    print(json.dumps(base_model.get_config(),indent=4 ))

# Add a callback to monitor layer outputs
class DebugCallback(tf.keras.callbacks.Callback):
    def on_predict_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, Activation):
                tf.print(f"\nLayer: {layer.name}")
                output = layer.output
                
                # Ensure output is a tensor before printing
                if isinstance(output, tf.Tensor):
                    tf.print("Output stats:",
                             "min:", tf.reduce_min(output),
                             "max:", tf.reduce_max(output),
                             "mean:", tf.reduce_mean(output),
                             "has_nan:", tf.reduce_any(tf.math.is_nan(output)))
                elif isinstance(output, KerasTensor):  # Check for KerasTensor
                    tf.print("Output is a KerasTensor, skipping. Type:", str(type(output)))  # Convert to string
                else:
                    # Handle the case where output is not a tensor or KerasTensor
                    tf.print("Output is not a tensor or KerasTensor, skipping. Type:", str(type(output)))  # Convert to string

    
# # test input
# input = np.random.rand(1, 224, 224, 3)
# input = (input - 0.5) * 2  # normalize to [-1, 1]

# local file
input_file = '/Users/matter/Downloads/gettyimages-1067956982.jpg.webp'
# input_file = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1280px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'
image = tf.keras.utils.load_img(input_file, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(image)  # Changed variable name from input to img_array

# # datasets      
# images = load_dataset(
#     'imagenet-1k',
#     split='train',
#     # streaming=True, 
#     use_auth_token=True
# )
# image = images[0]['image']
# img_array = np.array(image.resize((resolution, resolution)))

img_array = np.expand_dims(img_array, axis=0)  # coerce input shape to (1, 224, 224, 3)

img_array = preprocess_input(img_array)

result = base_model.predict(img_array, callbacks=[DebugCallback()]) if debug['print_layer_outputs'] else base_model.predict(img_array)

# print("\nFinal prediction shape:", result.shape)
# print("Contains NaN:", np.any(np.isnan(result)))
# print("Value range:", np.min(result), "to", np.max(result))
# print("\nSum of probabilities:", np.sum(result))

# Decode and print the top 5 predictions
predictions = decode_predictions(result, top=5)[0]
print("\nTop predictions:")
for pred in predictions:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")

# imshow
plt.imshow(image)
plt.axis('off')
plt.show()
    
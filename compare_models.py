import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import activations

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from ModelWrapper import ModelWrapper
from tiny_imagenet import TinyImagenetLoader
from ActivationFunction import (
    ActivationFunction, 
    ActivationFunctionFactory, 
    ApproximationType
)
from figlet_color import MOXIEPrettyPrint

def load_image_from_url(url, size=(224, 224)):
    """Load and preprocess an image from a URL"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize(size)
    return image

def compare_predictions(original_preds, modified_preds):
    """Compare and display predictions from both models"""
    print("\nPrediction Comparison:")
    print("-" * 50)
    print(f"{'Class':30} {'Original':>10} {'Modified':>10}")
    print("-" * 50)

    # Get the original model's top k predictions and labels
    orig_labels = original_preds['top_k_labels'][0]
    orig_probs = original_preds['top_k_probabilities'][0]
    
    # Get all probabilities from modified model
    mod_all_labels_dict = modified_preds['all_labels_dict']
    
    # For each top prediction from original model, get corresponding probability from modified model
    for i, label in enumerate(orig_labels):
        # Get original probability
        orig_prob = orig_probs[i]
        # Get corresponding probability from modified model
        mod_prob = mod_all_labels_dict.get(label, 0.0)
        
        print(f"{label:30} {orig_prob:10.4f} {mod_prob:10.4f}")

def main():
    # ----------------------------------------------------------------
    # Load test image
    image_url = "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg"
    image = load_image_from_url(image_url)

    # ----------------------------------------------------------------
    # Initialize original model
    print("Loading original model...")
    original_model = ModelWrapper(
        model_name="resnet50",
        model_type="image",
        model_class=ResNet50,
        processor_class=preprocess_input,
        is_huggingface=False
    )
    original_model.create_base_model()

    # ----------------------------------------------------------------
    # Initialize modified model
    print("\nLoading and modifying model with Chebyshev polynomial approximations...")
    modified_model = ModelWrapper(
        model_name="resnet50",
        model_type="image",
        model_class=ResNet50,
        processor_class=preprocess_input,
        is_huggingface=False
    )
    modified_model.create_base_model()
    
    # ----------------------------------------------------------------
    # Create activation function approximation
    factory = ActivationFunctionFactory(
        base_activation=activations.relu,
        degree=55,
        approximation_type=ApproximationType.CHEBYSHEV
    )
    chebyshev_activation = factory.create()
    
    # ----------------------------------------------------------------
    # Split and replace all activation layers
    modified_model.model = modified_model.split_activation_layers()
    # modified_model.replace_all_activations(activations.relu, chebyshev_activation)
    
    # Replace only the last ReLU activation layer
    relu_layers = [layer for layer in modified_model.model.layers if isinstance(layer, tf.keras.layers.Activation) and layer.activation.__name__ == "relu"]
    modified_model.replace_activation(relu_layers[-1], chebyshev_activation)

    # ----------------------------------------------------------------
    # Load training data using TinyImagenetLoader
    train_data, val_data, _ = TinyImagenetLoader.load_tiny_imagenet(root_dir="data/tiny-imagenet-200")
    train_images = train_data['images']
    train_labels = train_data['labels']
    val_images = val_data['images']
    val_labels = val_data['labels']
    
    # Retrain the modified model using the iterative method
    modified_model.retrain(train_images, train_labels)

    # Check accuracy on validation set
    val_acc = modified_model.model.evaluate(val_images, val_labels)[1]
    print(f"Validation accuracy: {val_acc:.4f}")

    # ----------------------------------------------------------------
    # Compile the modified model
    modified_model.model.compile()
    
    # ----------------------------------------------------------------
    # Get predictions from original model
    print("Getting predictions from original model...")
    original_preds = original_model.predict(image)

    # Get predictions from modified model
    print("Getting predictions from modified model...")
    modified_preds = modified_model.predict(image)

    # ----------------------------------------------------------------
    # Compare results
    compare_predictions(original_preds, modified_preds)
    compare_predictions(modified_preds, original_preds)

    # Display the test image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Test Image')
    plt.show()

if __name__ == "__main__":
    main() 
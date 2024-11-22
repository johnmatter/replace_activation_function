import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import activations

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from ModelWrapper import ModelWrapper
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

    # Get the pre-sorted top k predictions
    orig_probs = original_preds['top_k_probabilities'][0]
    mod_probs = modified_preds['top_k_probabilities'][0]
    orig_labels = original_preds['top_k_labels'][0]
    mod_labels = modified_preds['top_k_labels'][0]
    
    # Print top k predictions for both models
    for i in range(len(orig_labels)):
        print(f"{orig_labels[i]:30} {orig_probs[i]:10.4f} {mod_probs[i]:10.4f}")

def main():
    # ----------------------------------------------------------------
    # Test image
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
    
    # Create activation function approximation
    factory = ActivationFunctionFactory(
        base_activation=activations.sigmoid,
        degree=3,
        approximation_type=ApproximationType.CHEBYSHEV
    )
    chebyshev_activation = factory.create()
    
    # Split and replace activation layers
    modified_model.model = modified_model.split_activation_layers()
    modified_model.replace_activations(chebyshev_activation)
    
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

    # Display the test image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Test Image')
    plt.show()

if __name__ == "__main__":
    main() 
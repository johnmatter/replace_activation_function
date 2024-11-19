import tensorflow as tf
from ModelWrapper import ModelWrapper
from ActivationFunction import ActivationFunction
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from transformers import (
    TFViTForImageClassification,
    ViTImageProcessor,
    TFDeiTForImageClassification,
    DeiTFeatureExtractor,
    TFConvNextForImageClassification,
    ConvNextImageProcessor
)

def load_image_from_url(url):
    """Load and preprocess an image from a URL"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))  # Resize to expected size
    return image

def compare_predictions(original_preds, modified_preds, top_k=5):
    """Compare and display predictions from both models"""
    print("\nPrediction Comparison:")
    print("-" * 50)
    print(f"{'Class':30} {'Original':>10} {'Modified':>10}")
    print("-" * 50)
    
    # Get top k predictions for both models
    orig_probs = original_preds['probabilities'][0]
    mod_probs = modified_preds['probabilities'][0]
    
    # Get indices of top k predictions
    orig_indices = np.argsort(orig_probs)[-top_k:][::-1]
    mod_indices = np.argsort(mod_probs)[-top_k:][::-1]
    
    # Print top k predictions for both models
    for orig_idx, mod_idx in zip(orig_indices, mod_indices):
        orig_label = original_preds['predicted_labels'][0]
        mod_label = modified_preds['predicted_labels'][0]
        print(f"{orig_label:30} {orig_probs[orig_idx]:10.4f} {mod_probs[mod_idx]:10.4f}")

def main():
    # Test image URL
    image_url = "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg"
    
    # Load test image
    image = load_image_from_url(image_url)
    
    # Create debug configuration
    debug_config = {
        'print_network_split_debug': False,
        'print_activation_functions': False,
        'print_network_config': False,
        'print_layer_outputs': False,
        'print_layer_activations': False,
    }

    # Initialize original model (using ViT as an example)
    print("Loading original model...")
    original_model = ModelWrapper(
        model_name="google/vit-base-patch16-224",
        model_type="image",
        debug=debug_config,
        model_class=TFViTForImageClassification,
        processor_class=ViTImageProcessor
    )
    original_model.create_base_model()
    
    # Get predictions from original model
    print("Getting predictions from original model...")
    original_preds = original_model.predict(image)

    # Initialize modified model (using ViT as an example)
    print("\nLoading and modifying model with Taylor approximations...")
    modified_model = ModelWrapper(
        model_name="google/vit-base-patch16-224",
        model_type="image",
        debug=debug_config,
        model_class=TFViTForImageClassification,
        processor_class=ViTImageProcessor
    )
    modified_model.create_base_model()
    
    # Create activation function approximation
    taylor_activation = ActivationFunction(degree=3, piecewise=True)
    
    # Split and replace activation layers
    modified_model.model = modified_model.split_activation_layers()
    modified_model.replace_activations(taylor_activation)
    
    # Get predictions from modified model
    print("Getting predictions from modified model...")
    modified_preds = modified_model.predict(image)

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
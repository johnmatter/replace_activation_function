import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import activations

import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from ModelWrapper import ModelWrapper, RetrainType
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

    # Get the model's top k predictions and labels
    orig_labels = original_preds['top_k_labels'][0]
    orig_probs = original_preds['top_k_probabilities'][0]
    mod_labels = modified_preds['top_k_labels'][0]
    mod_probs = modified_preds['top_k_probabilities'][0]
    
    # For each top prediction from original model,
    # get corresponding probability from modified model
    for i, label in enumerate(orig_labels):
        if i > 10:
            break

        # Get original probability
        orig_prob = orig_probs[i]
        # Get corresponding probability from modified model
        label_idx = mod_labels.index(label) if label in mod_labels else -1
        mod_prob = mod_probs[label_idx] if label_idx != -1 else np.nan
        
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
        is_huggingface=False,
        input_shape=(224, 224, 3)
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
        is_huggingface=False,
        input_shape=(224, 224, 3)
    )
    modified_model.create_base_model()
    
    # ----------------------------------------------------------------
    # Create activation function approximation
    factory = ActivationFunctionFactory(
        base_activation=activations.relu,
        degree=5,
        approximation_type=ApproximationType.CHEBYSHEV
    )
    chebyshev_activation = factory.create()
    
    # ----------------------------------------------------------------
    # Split activation layers, set activation function
    modified_model.model = modified_model.split_activation_layers()
    modified_model.set_activation_function(chebyshev_activation)

    # ----------------------------------------------------------------
    # Load training data using TinyImagenetLoader
    print("\nLoading training data...")
    train_data, val_data, _ = TinyImagenetLoader.load_tiny_imagenet(root_dir="/Users/matter/Downloads/tiny-imagenet-200")
    
    # Take only 10000 samples for quick testing
    num_samples = 10000
    train_images = train_data['images'][:num_samples]
    train_labels = tf.keras.utils.to_categorical(train_data['labels'][:num_samples], num_classes=1000)
    
    # Take 1000 samples for validation
    val_images = val_data['images'][:1000]
    val_labels = tf.keras.utils.to_categorical(val_data['labels'][:1000], num_classes=1000)

    # Show example images to confirm data are loaded correctly
    # TinyImagenetLoader.show_example_images(train_data)
    
    # ----------------------------------------------------------------
    # Retrain the modified model
    modified_model.retrain(train_images, train_labels, RetrainType.ALL)

    # Save the modified model
    modified_model.save("modified_model.keras")

    # Check accuracy on validation set
    val_acc = modified_model.model.evaluate(val_images, val_labels)[1]
    print(f"Validation accuracy: {val_acc:.4f}")

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
import tensorflow as tf
from tensorflow.keras import activations
import matplotlib.pyplot as plt
import numpy as np
from ActivationFunction import ActivationFunctionFactory

def compare_activations():
    activation_functions = [
        activations.sigmoid,
        activations.relu,
        activations.tanh,
        # activations.softmax  # typically used for multi-class classification, and max is difficult in homomorphic encryption, so I'm skipping it
    ]
    
    degrees = np.arange(1, 20, 1)
    x_range = (-5, 5)
    num_points = 1000

    
    # Create a subplot grid for all activation functions
    num_activations = len(activation_functions)
    num_cols = 3  # Display 2 plots per row
    num_rows = (num_activations + num_cols - 1) // num_cols  # Ceiling division
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5*num_rows))
    axs = axs.flatten()  # Flatten the array of axes for easy indexing
    
    for idx, activation in enumerate(activation_functions):
        factory = ActivationFunctionFactory(base_activation=activation)
        fig = factory.compare_polynomial_approximations(degrees=degrees, x_range=x_range, num_points=num_points)
        fig.canvas.draw()  # Ensure the figure is drawn
        axs[idx].imshow(fig.canvas.buffer_rgba())  # Display the figure in the subplot
        axs[idx].axis('off')  # Hide the axes for a cleaner look
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_activations()
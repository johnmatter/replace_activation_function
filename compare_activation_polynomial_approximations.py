import tensorflow as tf
from tensorflow.keras import activations
import matplotlib.pyplot as plt
import numpy as np
from ActivationFunction import ActivationFunctionFactory, ApproximationType

output_dir = '/Users/matter/Downloads/activation_polynomial_approximations'

def compare_activations() -> None:
    """
    Compare polynomial approximations of different activation functions.
    Generates plots showing the original activation functions alongside their
    polynomial approximations of varying degrees, including residual plots.
    """
    activation_functions = [
        activations.sigmoid,
        activations.relu,
        activations.tanh,
        activations.gelu,
        activations.selu,
        activations.elu,
    ]
    
    degrees = np.arange(3, 20, 2)
    x_range = (-5, 5)
    num_points = 1000

    # Create a subplot grid for all activation functions
    num_activations = len(activation_functions)
    num_cols = 3
    num_rows = (num_activations + num_cols - 1) // num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5*num_rows))
    axs = axs.flatten()
    
    for approximation_type in ApproximationType:
        factory = ActivationFunctionFactory(approximation_type=approximation_type)
        output_path = f'{output_dir}/activation_polynomial_approximations_{approximation_type.value}.pdf'
        
        for idx, activation in enumerate(activation_functions):
            factory.base_activation = activation
            fig = factory.compare_polynomial_approximations(
                degrees=degrees, 
                x_range=x_range, 
                num_points=num_points, 
                debug=True
            )
            fig.canvas.draw()
            axs[idx].imshow(fig.canvas.buffer_rgba())
            axs[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
        # Save metrics DataFrame to CSV
        metrics_path = f'{output_dir}/activation_approximation_metrics_{approximation_type.value}.csv'
        factory.metrics_df.to_csv(metrics_path, index=False)
        
        # Plot metrics
        metrics_fig = factory.plot_metrics()
        metrics_fig_path = f'{output_dir}/activation_approximation_metrics_{approximation_type.value}.pdf'
        metrics_fig.savefig(metrics_fig_path, dpi=300)

if __name__ == "__main__":
    compare_activations()
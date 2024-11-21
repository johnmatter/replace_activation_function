import tensorflow as tf
from tensorflow.keras import activations
from scipy.interpolate import approximate_taylor_polynomial
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# For what it's worth, HEIR uses -0.004 * x^3 + 0.197 * x + 0.5 for sigmoid
# See https://github.com/google/heir/blob/f9e39963b863490eaec81e53b4b9d180a4603874/lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.cpp#L95-L100

class ActivationFunction:
    def __init__(self, poly):
        self.poly = poly

    def __call__(self, x):
        x = tf.clip_by_value(x, -5.0, 5.0)
        return sum([c * K.pow(x, i) for i, c in enumerate(self.poly.coefficients)])

class ActivationFunctionFactory:
    def __init__(self, base_activation=activations.sigmoid, degree=3):
        self.base_activation = base_activation
        self.degree = degree

    def _get_polynomial(self):
        if self.base_activation == activations.relu:
            # Use softplus as a smooth approximation of ReLU
            beta = 1.0
            return approximate_taylor_polynomial(
                lambda x: K.log(1 + K.exp(beta * x)) / beta,
                0,
                degree=self.degree,
                scale=1.0
            )
        elif self.base_activation == activations.sigmoid:
            return approximate_taylor_polynomial(
                self.base_activation, 
                0, 
                degree=self.degree, 
                scale=1.0
            )
        # Add more activation functions here
        else:
            return approximate_taylor_polynomial(
                self.base_activation, 
                0, 
                degree=self.degree, 
                scale=1.0
            )

    def create(self):
        poly = self._get_polynomial()
        return ActivationFunction(poly)    

    def compare_polynomial_approximations(self, degrees=[1,2,3,4,5], x_range=(-5,5), num_points=1000):
        """Compare polynomial approximations of different degrees to the original activation function."""
        # Create color scheme using viridis colormap
        num_colors = len(degrees)
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
        
        # Ensure x is a tensor
        x = tf.convert_to_tensor(np.linspace(x_range[0], x_range[1], num_points), dtype=tf.float32)
        
        # Create a new figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
        
        # Set y-axis to logarithmic scale for main plot
        ax1.set_yscale('log')
        
        # Get true activation values
        if hasattr(self.base_activation, '__call__'):
            if self.base_activation == activations.softmax:
                y_true = self.base_activation(tf.expand_dims(x, axis=-1))
            else:
                y_true = self.base_activation(x)
            
            ax1.plot(x, y_true, 'k-', label='Original', linewidth=2)
        
        # Plot approximations and residuals
        for i, degree in enumerate(degrees):
            factory = ActivationFunctionFactory(
                base_activation=self.base_activation,
                degree=degree
            )
            approx = factory.create()
            y_approx = approx(x)
            
            # Plot approximation
            ax1.plot(x, y_approx, 
                    color=colors[i],
                    linestyle=line_styles[i % len(line_styles)],
                    label=f'Degree {degree}',
                    alpha=0.8)
            
            # Plot residuals
            residuals = tf.abs(y_true - y_approx)
            ax2.plot(x, residuals,
                    color=colors[i],
                    linestyle=line_styles[i % len(line_styles)],
                    label=f'Degree {degree}',
                    alpha=0.8)
        
        # Configure main plot
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title(f'Polynomial Approximations of {self.base_activation.__name__}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Configure residuals plot
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for residuals too
        ax2.set_title('Absolute Residuals')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|y_true - y_approx|')
        
        plt.tight_layout()
        plt.close(fig)  # Close the figure to prevent display
        
        return fig
import tensorflow as tf
from tensorflow.keras import activations
from scipy.interpolate import approximate_taylor_polynomial
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import numpy.polynomial.chebyshev as chebyshev
from typing import List, Union, Callable, Tuple, Optional
import numpy.polynomial.legendre as legendre
from scipy.special import comb
from scipy.linalg import lstsq
from scipy.special import eval_legendre
from scipy.integrate import fixed_quad
import pdb

# For what it's worth, HEIR uses -0.004 * x^3 + 0.197 * x + 0.5 for sigmoid
# See https://github.com/google/heir/blob/f9e39963b863490eaec81e53b4b9d180a4603874/lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.cpp#L95-L100

class PolyWrapper:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

class ApproximationType(Enum):
    """Enum for different types of polynomial approximations."""
    TAYLOR = "taylor"
    CHEBYSHEV = "chebyshev"
    LEGENDRE = "legendre"
    CLENSHAW_CURTIS = "clenshaw_curtis"
    WEIGHTED_LEAST_SQUARES = "weighted_least_squares"
    BERNSTEIN = "bernstein"

class ActivationFunction:
    """
    A class representing an activation function approximated by a polynomial.
    
    Attributes:
        poly: A polynomial object with coefficients representing the approximation.
    """
    def __init__(self, poly: object, description: str) -> None:
        """
        Initialize the activation function with a polynomial approximation.
        
        Args:
            poly: Object containing coefficients representing the polynomial approximation.
        """
        self.poly = poly
        self.description = description
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply the polynomial approximation to the input tensor.
        
        Args:
            x: Input tensor to apply the activation function to.
            
        Returns:
            Tensor with the activation function applied.
        """
        x = tf.clip_by_value(x, -5.0, 5.0)
        return sum([c * K.pow(x, i) for i, c in enumerate(self.poly.coefficients)])

    def dump(self) -> None:
        # Print polynomial with lowest degree first, including placeholder variable x and degrees
        polynomial_str = ' + '.join([f'{c} * x^{i}' for i, c in enumerate(self.poly.coefficients)])
        dump_str = f'{self.description}: {polynomial_str}'
        print(dump_str)

class ActivationFunctionFactory:
    """
    Factory class for creating polynomial approximations of activation functions.
    """
    def __init__(
        self, 
        base_activation: Callable = activations.sigmoid, 
        degree: int = 3, 
        approximation_type: ApproximationType = ApproximationType.CHEBYSHEV,
        description: str = None,
        debug: bool = False
    ) -> None:
        """
        Initialize the factory with parameters for polynomial approximation.
        
        Args:
            base_activation: The activation function to approximate.
            degree: Maximum degree of the polynomial approximation.
            approximation_type: Type of polynomial approximation to use.
            description: Description of the activation function.
        """
        self.base_activation = base_activation
        self.degree = degree
        self.approximation_type = approximation_type
        self.description = description if description else self._get_description()
        self.debug = debug

    def _get_description(self) -> str:
        """
        Get the description of the activation function.
        
        Returns:
            Description of the activation function.
        """
        description = self.base_activation.__name__
        description += f' (approx. type: {self.approximation_type.value})'
        description += f' (degree {self.degree})'
        return description

    def _get_polynomial(self) -> object:
        """
        Get the polynomial approximation based on the specified type.
        
        Returns:
            A polynomial object with coefficients.
        """
        match self.approximation_type:
            case ApproximationType.TAYLOR:
                return self._get_taylor_polynomial()
            case ApproximationType.CHEBYSHEV:
                return self._get_chebyshev_polynomial()
            case ApproximationType.LEGENDRE:
                return self._get_legendre_polynomial()
            case ApproximationType.CLENSHAW_CURTIS:
                return self._get_clenshaw_curtis_polynomial()
            case ApproximationType.WEIGHTED_LEAST_SQUARES:
                return self._get_weighted_least_squares_polynomial()
            case ApproximationType.BERNSTEIN:
                return self._get_bernstein_polynomial()

    def _get_taylor_polynomial(self) -> object:
        """
        Compute Taylor series approximation of the activation function.
        
        Returns:
            Taylor series approximation object.
        """
        if self.base_activation == activations.relu:
            beta = 1.0
            return approximate_taylor_polynomial(
                lambda x: K.log(1 + K.exp(beta * x)) / beta,
                0,
                degree=self.degree,
                scale=1.0
            )
        else:
            return approximate_taylor_polynomial(
                self.base_activation, 
                0, 
                degree=self.degree, 
                scale=1.0
            )
    
    def _get_chebyshev_polynomial(self) -> object:
        """
        Compute Chebyshev polynomial approximation of the activation function.
        
        Returns:
            Chebyshev polynomial approximation object.
        """
        x_range = [-5.0, 5.0]
        num_points = 1000
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        if self.base_activation == activations.relu:
            beta = 1.0
            y = np.log(1 + np.exp(beta * x)) / beta
        else:
            y = self.base_activation(x).numpy()
        
        coeffs = chebyshev.chebfit(x, y, self.degree)
        power_series = chebyshev.cheb2poly(coeffs)
        
        return PolyWrapper(power_series)

    def _get_legendre_polynomial(self) -> object:
        """
        Compute Legendre polynomial approximation using mean-square approximation.
        
        Returns:
            Polynomial approximation object.
        """
        x_range = [-5.0, 5.0]
        num_points = 1000
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        if self.base_activation == activations.relu:
            beta = 1.0
            y = np.log(1 + np.exp(beta * x)) / beta
        else:
            y = self.base_activation(x).numpy()
        
        # Compute Legendre coefficients
        coeffs = legendre.legfit(x, y, self.degree)
        power_series = legendre.leg2poly(coeffs)
        
        return PolyWrapper(power_series)

    def _get_clenshaw_curtis_polynomial(self) -> object:
        """
        Compute polynomial approximation using Clenshaw-Curtis quadrature.
        
        Returns:
            Polynomial approximation object.
        """
        def chebyshev_points(n):
            return np.cos(np.pi * np.arange(n) / (n - 1))
        
        # Map [-1, 1] to [-5, 5]
        scale = 5.0
        n_points = self.degree + 1
        x_cheb = scale * chebyshev_points(n_points)
        
        if self.base_activation == activations.relu:
            beta = 1.0
            y_cheb = np.log(1 + np.exp(beta * x_cheb)) / beta
        else:
            y_cheb = self.base_activation(x_cheb).numpy()
        
        # Compute coefficients using barycentric interpolation
        def barycentric_weights(x):
            n = len(x)
            w = np.ones(n)
            for j in range(n):
                for k in range(n):
                    if k != j:
                        w[j] *= 2.0 / (x[j] - x[k])
            return w
        
        weights = barycentric_weights(x_cheb)
        
        # Convert to power series
        x_eval = np.linspace(-scale, scale, 1000)
        y_eval = np.zeros_like(x_eval)
        
        for i, x in enumerate(x_eval):
            if x in x_cheb:
                y_eval[i] = y_cheb[np.where(x_cheb == x)[0][0]]
            else:
                numer = sum(w * y / (x - x_cheb) for w, x_cheb, y in zip(weights, x_cheb, y_cheb))
                denom = sum(w / (x - x_cheb) for w, x_cheb in zip(weights, x_cheb))
                y_eval[i] = numer / denom
        
        # Fit polynomial to the evaluated points
        coeffs = np.polyfit(x_eval, y_eval, self.degree)
        
        return PolyWrapper(coeffs[::-1])

    def _get_weighted_least_squares_polynomial(self) -> object:
        """
        Compute polynomial approximation using weighted least squares.
        
        Returns:
            Polynomial approximation object.
        """
        x_range = [-5.0, 5.0]
        num_points = 1000
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        if self.base_activation == activations.relu:
            beta = 1.0
            y = np.log(1 + np.exp(beta * x)) / beta
        else:
            y = self.base_activation(x).numpy()
        
        # Create weight function (giving more weight to points near 0)
        weights = np.exp(-0.1 * np.abs(x))
        
        # Create Vandermonde matrix
        A = np.vander(x, self.degree + 1)
        
        # Apply weights
        weighted_A = A * weights[:, np.newaxis]
        weighted_y = y * weights
        
        # Solve weighted least squares problem
        coeffs, *_ = lstsq(weighted_A, weighted_y)
        
        return PolyWrapper(coeffs[::-1])

    def _get_bernstein_polynomial(self) -> object:
        """
        Compute Bernstein polynomial approximation.
        
        Returns:
            Polynomial approximation object.
        """
        x_range = [-5.0, 5.0]
        num_points = 1000
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        if self.base_activation == activations.relu:
            beta = 1.0
            y = np.log(1 + np.exp(beta * x)) / beta
        else:
            y = self.base_activation(x).numpy()
        
        # Scale x to [0,1] for Bernstein polynomials
        x_scaled = (x - x_range[0]) / (x_range[1] - x_range[0])
        
        def bernstein_basis(i, n, t):
            return comb(n, i) * (t**i) * ((1-t)**(n-i))
        
        # Compute Bernstein coefficients
        n = self.degree
        coeffs = np.zeros(n + 1)
        
        for i in range(n + 1):
            t = i / n
            x_t = t * (x_range[1] - x_range[0]) + x_range[0]
            if self.base_activation == activations.relu:
                coeffs[i] = np.log(1 + np.exp(beta * x_t)) / beta
            else:
                coeffs[i] = self.base_activation(tf.constant(x_t)).numpy()
        
        # Convert Bernstein form to power basis
        power_coeffs = np.zeros(n + 1)
        for i in range(n + 1):
            for j in range(i + 1):
                power_coeffs[j] += coeffs[i] * comb(n, i) * comb(i, j) * (-1)**(i-j)
        
        # Scale coefficients back to original domain
        scaled_coeffs = np.zeros(n + 1)
        scale = x_range[1] - x_range[0]
        shift = x_range[0]
        
        for i in range(n + 1):
            for j in range(i + 1):
                scaled_coeffs[j] += power_coeffs[i] * comb(i, j) * (shift**(i-j)) / (scale**i)
        
        return PolyWrapper(scaled_coeffs)

    def create(self):
        poly = self._get_polynomial()
        return ActivationFunction(poly, self.description)    

    def compare_polynomial_approximations(
        self, 
        degrees: List[int] = [1,2,3,4,5], 
        x_range: Tuple[float, float] = (-5,5), 
        num_points: int = 1000,
        log_scale: bool = False,
        debug: bool = False
    ) -> plt.Figure:
        """
        Compare polynomial approximations of different degrees to the original activation function.
        
        Args:
            degrees: List of polynomial degrees to compare.
            x_range: Tuple of (min, max) x values for evaluation.
            num_points: Number of points to evaluate functions at.
        
        Returns:
            matplotlib Figure containing the comparison plots.
        """
        # Create color scheme using viridis colormap
        num_colors = len(degrees)
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
        
        # Ensure x is a tensor
        x = tf.convert_to_tensor(np.linspace(x_range[0], x_range[1], num_points), dtype=tf.float32)
        
        # Create a new figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[2, 1])
        
        # Set y-axis to logarithmic scale for main plot
        if log_scale:
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
                approximation_type=self.approximation_type,
                degree=degree
            )
            approx = factory.create()
            if debug:
                approx.dump()
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
        if log_scale:
            ax2.set_yscale('log')  # Log scale for residuals too
        ax2.set_title('Absolute Residuals')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|y_true - y_approx|')
        
        plt.tight_layout()
        plt.close(fig)  # Close the figure to prevent display
        
        return fig

import tensorflow as tf
from tensorflow.keras import activations
from scipy.interpolate import approximate_taylor_polynomial
from tensorflow.keras import backend as K

class ActivationFunction:
    def __init__(self, base_activation=activations.sigmoid, degree=3, piecewise=False):
        self.base_activation = base_activation
        self.degree = degree
        self.piecewise = piecewise
        self._initialize_polynomials()

    def _initialize_polynomials(self):
        if not self.piecewise:
            # For what it's worth, HEIR uses -0.004 * x^3 + 0.197 * x + 0.5 for sigmoid
            # See https://github.com/google/heir/blob/f9e39963b863490eaec81e53b4b9d180a4603874/lib/Dialect/TOSA/Conversions/TosaToSecretArith/TosaToSecretArith.cpp#L95-L100
            self.poly = approximate_taylor_polynomial(
                self.base_activation, 
                0, 
                degree=self.degree, 
                scale=1.0
            )
        else:
            self.poly_neg = approximate_taylor_polynomial(
                self.base_activation, 
                -1, 
                degree=self.degree, 
                scale=0.5
            )
            self.poly_zero = approximate_taylor_polynomial(
                self.base_activation, 
                0, 
                degree=self.degree, 
                scale=0.5
            )
            self.poly_pos = approximate_taylor_polynomial(
                self.base_activation, 
                1, 
                degree=self.degree, 
                scale=0.5
            )

    def __call__(self, x):
        x = tf.clip_by_value(x, -5.0, 5.0)
        
        if not self.piecewise:
            return sum([c * K.pow(x, i) for i, c in enumerate(self.poly.coefficients)])
        
        return K.switch(x < -0.5,
                    sum([c * K.pow(x, i) for i, c in enumerate(self.poly_neg.coefficients)]),
                    K.switch(x > 0.5,
                            sum([c * K.pow(x, i) for i, c in enumerate(self.poly_pos.coefficients)]),
                            sum([c * K.pow(x, i) for i, c in enumerate(self.poly_zero.coefficients)])))

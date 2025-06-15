import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int,
                 activation_function: callable):
        self.weights = np.ones((n_inputs, n_neurons))
        self.bias = np.ones((1, n_neurons))
        self.activation_function = activation_function

    def forward(self, X):
        linear_combination_with_weights = X @ self.weights + self.bias
        return self.activation_function(linear_combination_with_weights)
    
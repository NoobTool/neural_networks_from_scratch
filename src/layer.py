import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int,
                 activation_function: callable,
                 derivative_activation_function: callable):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.bias = np.ones((1, n_neurons))
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

        # Storing for backpropagation.
        self.input = None # Input provided to the layer.
        self.z = None # The result of matrix multiplication of weights & input.
        self.output = None # The output after applying activation function.

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.input = X  # store for backprop
        self.z = X @ self.weights + self.bias
        self.output = self.activation_function(self.z)
        return self.output
    
    def backward(self, derivative_of_next: np.array, learning_rate: float):
        derivative_of_this_layer = derivative_of_next * self.derivative_activation_function(self.z)
        derivative_of_this_weights = self.input.T @ derivative_of_this_layer
        derivative_of_this_bias = np.sum(derivative_of_this_layer, axis=0, keepdims=True)

        self.weights -= learning_rate * derivative_of_this_weights
        self.bias -= learning_rate * derivative_of_this_bias

        return derivative_of_this_layer @ self.weights.T
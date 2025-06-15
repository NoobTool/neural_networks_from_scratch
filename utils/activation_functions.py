import numpy as np

class ActivationFunctions:
    """
    A collection of common activation functions for neural networks, each with formula and ASCII-graph.
    """

    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation.
        Formula: f(x) = max(0, x)
        """
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation.
        Formula: f(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation without using np.tanh.
        Formula: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        """
        # Compute tanh manually for each element
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)

    @staticmethod
    def softmax(x):
        """
        Softmax activation.
        Formula: f_i(x) = exp(x_i) / sum_j exp(x_j)
        """
        # Shift inputs for numerical stability
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

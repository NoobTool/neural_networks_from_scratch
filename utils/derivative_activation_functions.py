import numpy as np

class DerivativeActivationFunctions:
    @staticmethod
    def relu_derivative(x):
        """
        Derivative of ReLU activation:
        f(x) = max(0, x)
        f'(x) = 1 if x > 0 else 0
        """
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of Sigmoid activation:
        f(x) = 1 / (1 + exp(-x))
        f'(x) = f(x) * (1 - f(x))
        """
        sig = DerivativeActivationFunctions.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def tanh(x):
        e_pos = np.exp(x)
        e_neg = np.exp(-x)
        return (e_pos - e_neg) / (e_pos + e_neg)

    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of Tanh activation:
        f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        f'(x) = 1 - f(x)^2
        """
        t = DerivativeActivationFunctions.tanh(x)
        return 1 - t ** 2

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def softmax_derivative(x):
        """
        Approximate element-wise derivative of Softmax:
        f_i(x) = exp(x_i) / sum_j exp(x_j)
        f'_i(x) ≈ f_i(x) * (1 - f_i(x))

        Note: This is NOT the true Jacobian. Use y_pred - y_true for softmax + cross-entropy.
        """
        s = DerivativeActivationFunctions.softmax(x)
        return s * (1 - s)

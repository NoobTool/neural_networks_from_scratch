import numpy as np

from src.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer], 
                 learning_rate: float):
        self.layers = layers
        self.learning_rate = learning_rate

    def train(self, x_train: np.array, y_train: np.array,
              n_epochs: int):
        
        for epoch in range(n_epochs):

            # Forward pass through all layers
            output = x_train
            for layer in self.layers:
                output = layer.forward(output)

            # Compute derivative of loss (MSE)
            loss = np.mean((output - y_train) ** 2)
            d_loss = output - y_train

            # Backward pass through layers in reverse
            gradient = d_loss
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.learning_rate)

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss:.6f}")
    
    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def infer(data: np.array):
        pass
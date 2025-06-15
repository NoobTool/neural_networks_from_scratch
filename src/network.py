import numpy as np

from src.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def train(self, x_train: np.array, y_train: np.array,
              n_epochs: int):
        for i in range(n_epochs):

            output = x_train
            for layer in self.layers:
                output = layer.forward(output)
        
        return output

    def infer(data: np.array):
        pass
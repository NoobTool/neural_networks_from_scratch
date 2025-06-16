🧠 Neural Network from Scratch – MNIST Digit Classifier
![MNIST Data Sample](https://github.com/user-attachments/assets/50384621-b461-499e-9dba-8741125cb65f)

🔍 Overview
This project demonstrates a neural network built entirely from scratch using NumPy, trained on the MNIST digit dataset (8x8 version from sklearn.datasets), achieving over 91% accuracy.

No frameworks like TensorFlow, PyTorch, or Keras were used. Every layer, activation function, and backpropagation step is manually implemented to gain a deep understanding of how neural networks work under the hood.

📁 Directory Structure
<pre>
.
├── nn_from_scratch.ipynb               # Jupyter Notebook to train and test the model
├── requirements.txt                    # Required Python packages
├── src
│   ├── layer.py                        # Layer class with forward/backward propagation
│   └── network.py                      # NeuralNetwork class combining layers
└── utils
    ├── activation_functions.py         # ReLU, Sigmoid, Tanh, Softmax implementations
    └── derivative_activation_functions.py  # Corresponding activation derivatives
</pre>
  
🚀 Features

- Custom forward pass and backpropagation using OOPS.
- Support for ReLU, Sigmoid, Tanh, and Softmax activations.
- Achieves >91% accuracy on MNIST digits.
- Includes visualization of sample images for manual inspection.

🧠 Core Concepts Reinforced
- Forward and backward propagation
- Derivatives of activation functions
- Weight initialization (He Initialization)
- Debugging numerical instabilities

🧪 How to Run
- Clone the repo:
- git clone https://github.com/NoobTool/neural_networks_from_scratch.git
- cd neural_networks_from_scratch
- Install dependencies: pip install -r requirements.txt
- Run the notebook:
  - Launch nn_from_scratch.ipynb in Jupyter Notebook or Jupyter Lab:
    - jupyter notebook

📈 Results
| Metric    | Value       |
|-----------|-------------|
| Accuracy  | 91.39%      |
| Dataset   | MNIST (8x8) |
| Training  | 100 epochs  |


🙌 Acknowledgments
Inspired by the need to understand how neural networks operate at the mathematical level.
Built for personal learning, with the hope it helps others demystify deep learning.

📬 Connect
If you’ve tried something similar, or want to dive deeper into model internals, feel free to connect on LinkedIn or contribute to the project!


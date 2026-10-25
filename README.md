# Neural Network from Scratch

A compact, educational implementation of a neural network built using only NumPy. The project demonstrates how a network can be trained from scratch to approximate simple functions such as sine, cosine, and logical gates.

## Overview

This repository contains a minimal neural network implementation with:

- forward propagation
- backpropagation
- gradient descent training
- configurable hidden layers and activation functions
- synthetic dataset generators
- a plotting example to visualize model behavior before and after training

## Project Structure

- `neural_network.py` — core neural network implementation
- `data_generator.py` — synthetic data generation for regression and logic tasks
- `main.py` — example training script and visualization
- `README.md` — project overview and usage instructions

## Network Implementation

The `NN` class supports:

- custom layer sizes
- activation functions: `sigmoid` and `tanh`
- weight and bias updates through gradient descent
- single-output and multi-output training patterns

Training is done manually using NumPy operations, which makes the code easy to follow and modify for learning purposes.

## Included Data Generators

The `generate` class includes example datasets for:

- `sin(data_size)`
- `cos(data_size)`
- `sincos(data_size)`
- `and_gate()`
- `or_gate()`

These are useful for experimenting with different network architectures and learning tasks.

## Example Usage

```python
import numpy as np
from neural_network import NN
from data_generator import generate

X, y = generate.sin(25)

network = NN([1, 20, 1], 'tanh')

for _ in range(5000):
    network.train(X, y, learning_rate=0.02, batch_size=1)

predictions = network.forward_pass(X, is_training=False)
print(predictions)
```

## Run the Example

1. Install dependencies:

```bash
pip install numpy matplotlib
```

2. Run the training demo:

```bash
python main.py
```

This script trains the network on a generated sine wave and displays the pre-training and post-training outputs in a plot.

## Notes

- This project is intentionally simple and designed for learning.
- It is a good starting point for understanding how neural networks are built from fundamental math and linear algebra.
- You can easily modify the dataset, number of layers, activation function, and learning rate to experiment with different behaviors.

## License

This project is provided as-is for educational use.

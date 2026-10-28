import numpy as np
import matplotlib.pyplot as plt
from src.nn_framework import NN
from src.data_generator import generate

# Network Attributes:
network_structure = [2, 4, 1]
activation_function_type = 'sigmoid'

# Network Training Parameters:
simulated_function_type = 'andgate'
epochs = 2000
learning_rate = 0.2
batch_size = 1

# Initialization
nn = NN(network_structure, activation_function_type)
pre_training_output = nn.forward_pass(data[0], is_training=False)

# Data Generation:
data = generator.generate(simulated_function_type, data_size)
shuffeled_indices = np.arange(data[0].shape[0])
np.random.seed(42)
np.random.shuffle(shuffeled_indices)

# Training
for epoch in range(epochs):
	nn.train(data[0][shuffeled_indices], data[1][shuffeled_indices], learning_rate, batch_size)

# Testing Network
post_training_output = nn.forward_pass(data[0], is_training=False)
print('Before Training')
for i in range(len(data[0])):
	print(data[0][i], data[1][i], pre_training_output[i])

print('After Training')
for i in range(len(data[0])):
	print(data[0][i], data[1][i], post_training_output[i])

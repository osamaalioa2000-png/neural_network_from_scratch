import numpy as np
import matplotlib.pyplot as plt
from src.nn_framework import NN
from src.data_generator import generate

# Network Attributes
network_structure = [1, 20, 1]
activation_function_type = 'tanh'

# Network Training Parameters
simulated_function_type = 'sin'
epochs = 3000
learning_rate = 0.02
batch_size = 1
data_size = 64

# Initialization
nn = NN(network_structure, activation_function_type)

# Data Generation
data = generator.generate(simulated_function_type, data_size)
shuffeled_indices = np.arange(data[0].shape[0])
np.random.seed(42)
np.random.shuffle(shuffeled_indices)

test_data_size = 256
test_data = generator.generate(simulated_function_type, test_data_size)
pre_training_output = nn.forward_pass(test_data[0], is_training=False)

# Training
for epoch in range(epochs):
	nn.train(data[0][shuffeled_indices], data[1][shuffeled_indices], learning_rate, batch_size)
	
# Testing Network
post_training_output = nn.forward_pass(test_data[0], is_training=False)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].plot(test_data[0], test_data[1])
axes[0].plot(test_data[0], pre_training_output)
axes[0].legend(['Target Output', 'Network Output'])
axes[0].set_title('Before Training')

axes[1].plot(test_data[0], test_data[1])
axes[1].plot(test_data[0], post_training_output)
axes[1].legend(['Target Output', 'Network Output'])
axes[1].set_title('After Training')

plt.tight_layout()
plt.show()

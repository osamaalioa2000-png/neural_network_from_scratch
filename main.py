import numpy as np
from neural_network_github import NN
# from final_github import NN
from data_generator_github import generate
import matplotlib.pyplot as plt

# Network Attributes:
network_structure = [1, 20, 1]
activation_function_type = 'tanh'

# Network Training Parameters:
epochs = 5000
learning_rate = 0.02
batch_size = 1
data_size = 25
training_data = 20

# Data Generation:
data = generate.sin(data_size)
# data = generate.cos(data_size)
# data = generate.sincos(data_size)
# data = generate.and_gate()
# data = generate.or_gate()
shuffeled_indices = np.arange(data[0].shape[0])
np.random.seed(42)
np.random.shuffle(shuffeled_indices)

# Initialization
nn = NN(network_structure, activation_function_type)
pre_training_output = nn.forward_pass(np.sort(data[0]), is_training=False)

# Training
for epoch in range(epochs):
	nn.train(data[0][shuffeled_indices][:training_data], data[1][shuffeled_indices][:training_data], learning_rate, batch_size)
	
post_training_output = nn.forward_pass(data[0], is_training=False)

# Visualization
fig, axes = plt.subplots(1, 2)

axes[0].plot(data[0], data[1])
axes[0].plot(data[0], pre_training_output)
axes[0].legend(['Target Output', 'Network Output'])
axes[0].set_title('Pre-Training Output')

axes[1].plot(data[0], data[1])
axes[1].plot(data[0], post_training_output)
axes[1].legend(['Target Output', 'Network Output'])
axes[1].set_title('Post-Training Output')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.nn_framework import NN
from src.data_generator import generator

# Network Attributes:
network_structure = [2, 32, 32, 1]
activation_function_type = 'tanh'

# Network Training Parameters:
simulated_function_type = 'sin*cos'
epochs = 10000
learning_rate = 0.005
batch_size = 1
data_size = 64

# Network Initialization
nn = NN(network_structure, activation_function_type)

# Data Generation and Shuffling:
data = generator.generate(simulated_function_type, data_size)
shuffeled_indices = np.arange(data[0].shape[0])
np.random.seed(42)
np.random.shuffle(shuffeled_indices)

test_data_size = 256
test_data = generator.generate(simulated_function_type, test_data_size)
pre_training_output = nn.forward_pass(test_data[0], is_training=False)

# Training
for epoch in range(epochs):
	nn.train(data[0], data[1], learning_rate, batch_size)
	

# Validation
post_training_output = nn.forward_pass(test_data[0], is_training=False)

# Visualization
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(4, 3, figure=fig)

n = int(np.sqrt(test_data_size))
x = test_data[0][:, 0].reshape(n, n)
y = test_data[0][:, 1].reshape(n, n)

axes = [[], [], []]
axes[0].append(fig.add_subplot(gs[0:2, 0], projection='3d'))
axes[0].append(fig.add_subplot(gs[0:2, 1], projection='3d'))
axes[1].append(fig.add_subplot(gs[2:4, 0], projection='3d'))
axes[1].append(fig.add_subplot(gs[2:4, 1], projection='3d'))
axes[2].append(fig.add_subplot(gs[1:3, 2], projection='3d'))

z_target = test_data[1].reshape(n, n)
axes[1][0].set_title('Target Output')
axes[1][0].plot_surface(x, y, z_target)

z_network_pre = pre_training_output.reshape(n, n)
axes[0][0].set_title('Network Output Before Training')
axes[0][0].plot_surface(x, y, z_network_pre)

z_target = test_data[1].reshape(n, n)
axes[1][1].set_title('Target Output')
axes[1][1].plot_surface(x, y, z_target)

z_network_post = post_training_output.reshape(n, n)
axes[0][1].set_title('Network Output After Training')
axes[0][1].plot_surface(x, y, z_network_post)

axes[2][0].plot_surface(x, y, z_target - z_network_post)
axes[2][0].set_title('Post-Training Error Surface')
axes[2][0].set_zlim(np.min(data[1]), np.max(data[1]))

plt.tight_layout()
plt.show()

import numpy as np

class NN:
	def __init__(self, layer_sizes):
		self.layer_num = len(layer_sizes)
		self.layer_sizes = layer_sizes
		rng = np.random.default_rng(42)
		self.weights = [rng.random((layer_sizes[i], layer_sizes[i + 1])) for i in range(self.layer_num - 1)]
		self.biases = [rng.random(layer_sizes[i]) for i in range(1, self.layer_num)]
		
	def activation_function(self, signals, function_type):
		if function_type == 'sigmoid':
			return 1 / (1 + np.exp(-signals))
			
		if function_type == 'linear':
			return signals
			
		if function_type == 'tanh':
			return (np.exp(signals) - np.exp(-signals)) / (np.exp(signals) + np.exp(-signals))
			
		
	def forward_pass(self, input_layer, is_training=True):
		current_layer = input_layer
		if is_training:
			self.signals = [input_layer]
			
		for i in range(self.layer_num - 1):
			current_layer = self.activation_function(current_layer @ self.weights[i] + self.biases[i], 'sigmoid')
			if is_training:
				self.signals.append(current_layer)
				
			
		return current_layer
		
	def gradient(self, actual_output, target_output):
		delta = np.array((actual_output - target_output) * actual_output * (1 - actual_output), ndmin=2).T
		weight_grad = [np.einsum('ik,ij->kj', np.array(delta, ndmin=2), np.array(self.signals[-2], ndmin=2)).T]
		bias_grad = [np.sum(delta, axis=0)]
		for i in range(self.layer_num)[-2:0:-1]:
			delta = np.array((delta @ self.weights[i].T) * self.signals[i] * (1 - self.signals[i]), ndmin=2)
			weight_grad.append(np.einsum('ik,ij->kj', np.array(delta, ndmin=2), np.array(self.signals[i - 1], ndmin=2)).T)
			bias_grad.append(np.sum(delta, axis=0))
			
		return weight_grad[::-1], bias_grad[::-1]
		
	def descent(self, weight_grad, bias_grad, learning_rate):
		for i in range(self.layer_num - 1):
			self.weights[i] -= learning_rate * weight_grad[i]
			self.biases[i] -= learning_rate * bias_grad[i]
			
		
	def train(self, input_data, output_data, learning_rate, batch_size=1):
		low_bound = 0
		for high_bound in list(range(len(input_data)))[batch_size::batch_size] + [len(input_data)]:
			network_output = self.forward_pass(input_data[low_bound:high_bound], is_training=True)
			grad = self.gradient(network_output, output_data[low_bound:high_bound])
			self.descent(grad[0], grad[1], learning_rate)
			low_bound = high_bound
			
		
	

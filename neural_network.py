import numpy as np

class NN:
	def __init__(self, layer_sizes):
		self.layer_sizes = layer_sizes
		rng = np.random.default_rng(42)
		self.weights = [rng.random((layer_sizes[i + 1], layer_sizes[i])) for i in range(len(self.layer_sizes) - 1)]
		self.biases = [[rng.random(self.layer_sizes[i])] for i in range(1, len(self.layer_sizes))]
		
	def activation_function(self, signals, function_type):
		if function_type == 'sigmoid':
			return 1 / (1 + np.exp(-signals))
			
		if function_type == 'linear':
			return signals
			
		if function_type == 'tanh':
			return (np.exp(signals) - np.exp(-signals)) / (np.exp(signals) + np.exp(-signals))
			
		
	def run_input(self, input_layer, is_training=True):
		current_layer = input_layer
		self.all_signals = [input_layer]
		for i in range(len(self.layer_sizes) - 1):
			current_layer = self.activation_function(self.weights[i] @ current_layer + np.transpose(self.biases[i]), 'tanh')
			if is_training:
				self.all_signals.append(current_layer)
				
			
		return current_layer
		
	def gradient(self, target_output, actual_output):
		delta = ((actual_output - target_output) * (1 - actual_output ** 2))
		weight_grad = [delta @ np.transpose(self.all_signals[-2])]
		bias_grad = [delta]
		for i in range(len(self.weights) - 1)[::-1]:
			delta = (delta @ self.weights[i + 1]) * (1 - np.transpose(self.all_signals[i + 1]) ** 2)
			weight_grad.insert(0, np.transpose(delta) @ np.transpose(self.all_signals[i]))
			bias_grad.insert(0, delta)
			
		return weight_grad, bias_grad
		
	def descent(self, weight_grad, bias_grad, learning_rate):
		for i in range(len(self.weights)):
			self.weights[i] -= learning_rate * weight_grad[i]
			self.biases[i] -= learning_rate * bias_grad[i]
			
		
	def train(self, training_data, learning_rate):
		for data in training_data:
			for input_data, target_output in data:
				output = self.run_input(input_data, is_training=True)
				grad = self.gradient(target_output, output)
				self.descent(grad[0], grad[1], learning_rate)
			
		
	

import numpy as np

class generate:
	def sin(data_size):
		input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
		output_vec = np.sin(input_vec)
		return input_vec, output_vec
		
	def cos(data_size):
		input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
		output_vec = np.cos(input_vec)
		return input_vec, output_vec
		
	def sincos(data_size):
		input_vec = np.linspace(-2 * np.pi, 2 * np.pi, data_size)
		input_mat = np.array([input_vec, input_vec]).T
		output_vec = np.array(np.sin(input_vec) * np.cos(input_vec), ndmin=2).T
		return input_mat, output_vec

	def and_gate():
		input_vec = np.array([
							[0, 0],
							[0, 1],
							[1, 0],
							[1, 1]], ndmin=2)
			
		output_vec = np.array([
							[0],
							[0],
							[0],
							[1]], ndmin=2)
		
		return input_vec, output_vec
		
	def or_gate():
		input_vec = np.array([
							[0, 0],
							[0, 1],
							[1, 0],
							[1, 1]], ndmin=2)
		
		output_vec = np.array([
							[0],
							[1],
							[1],
							[1]], ndmin=2)
		
		return input_vec, output_vec
		
	

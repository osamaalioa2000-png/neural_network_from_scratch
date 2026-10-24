import numpy as np

class generate:
	def sin(data_size):
		input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
		return input_vec, np.sin(input_vec)
		
	def cos(data_size):
		input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
		return input_vec, np.cos(input_vec)
		
	def andgate():
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
		
	def orgate():
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
		
	

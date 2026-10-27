import numpy as np

class generator:
	def generate(func_type, data_size):
		if func_type == 'sin':
			input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
			output_vec = np.sin(input_vec)
			return input_vec, output_vec
			
		elif func_type == 'cos':
			input_vec = np.array(np.linspace(-2 * np.pi, 2 * np.pi, data_size), ndmin=2).T
			output_vec = np.cos(input_vec)
			return input_vec, output_vec
			
		elif func_type.replace(' ', '') == 'sin*cos':
			input_vec = np.linspace(-np.pi, np.pi, int(np.sqrt(data_size)))
			x, y = np.meshgrid(input_vec, input_vec)
			input_mat = np.column_stack([x.ravel(), y.ravel()])
			output_vec = np.array(np.sin(input_mat[:, 0]) * np.cos(input_mat[:, 1]), ndmin=2).T
			return input_mat, output_vec

		elif func_type.replace(' ', '') == 'andgate':
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
			
		elif func_type.replace(' ', '') == 'orgate':
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
			
		else:
			print('Simulated Function Type Not Supported')
			quit()
			
		
	

import numpy as np
import math

class generate:
	def sincos(data_size):
		input_vector1 = np.linspace(-2 * np.pi, 2 * np.pi, int(math.sqrt(data_size)))
		input_vector2 = np.linspace(-2 * np.pi, 2 * np.pi, int(math.sqrt(data_size)))
		training_data = []
		for i in range(int(math.sqrt(data_size))):
			training_data.append([])
			for j in range(int(math.sqrt(data_size))):
				training_data[i].append([np.array([[input_vector1[i]], [input_vector2[j]]], ndmin=2), np.array(np.sin(input_vector1[i]) * np.cos(input_vector2[j]), ndmin=2)])
				
			
		rng = np.random.default_rng(42)
		rng.shuffle(training_data)
		return training_data
		
	def sin(data_size):
		input_vector = np.linspace(-2 * np.pi, 2 * np.pi, data_size)
		training_data = []
		for i in range(data_size):
			training_data.append([np.array([[input_vector[i]]], ndmin=2), np.array(np.sin(input_vector[i]), ndmin=2)])
			
		rng = np.random.default_rng(42)
		rng.shuffle(training_data)
		return [training_data]
		
	

# training_data = [[[np.array([[0], [0]]), np.array(0)],
				 #  [np.array([[0], [1]]), np.array(0)]],
				 # [[np.array([[1], [0]]), np.array(0)],
				 #  [np.array([[1], [1]]), np.array(1)]]]

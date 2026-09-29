import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text
import backup
from data_generator import training_data

nn = backup.NN([2, 10, 10, 1])
# nn = backup.NN([2, 10, 1])

epochs = 2000
learning_rate = 0.03
avgerrors = []
for epoch in range(epochs):
	nn.train(training_data, learning_rate)
	
avg_error = 0
for x in training_data:
	for data in x:
		avg_error += abs(nn.run_input(data[0]) - data[1]) / 2 * 100
		
	
avg_error /= len(training_data) ** 2
print(avg_error[0][0])

# outputs = []
# errors = []
# sumerror = 0
# for x in range(len(training_data[:40])):
	# outputs.append([])
	# errors.append([])
	# for y in range(len(training_data[x])):
		# outputs[x].append(nn.run_input(training_data[x][y][0], is_training=False)[0][0])
		# errors[x].append(abs(outputs[x][y] - training_data[x][y][1][0][0]))
		# sumerror += abs(errors[x][y])
		
	
# avgerrors.append(sumerror / (len(errors) * len(errors[0])) / 2 * 100)
# if len(avgerrors) > 51 and max(avgerrors[-51:-1]) - min(avgerrors[-51:-1]) < 0.5:
	# print(epoch)
	# break


# fig, ax = plt.subplots()
# plt.text(-1, 50, 'Average Error: {:.4f}'.format(avgerrors[-1]))
# ax.plot(list(range(len(avgerrors))), avgerrors, linewidth=1)
# plt.show()

# sin(x) training data minimum error: 0.7462%, learning rate: 0.02, epochs: 5000, 1 hidden layer, hidden neurons: 30
# sin(x) test data error: 0.8656%, learning rate: 0.02, epochs: 5000, 1 hidden layer, hidden neurons: 30
# sin(x) training data minimum error: 1.0258%, learning rate: 0.02, epochs: 1000, 2 hidden layers, hidden neurons: 5, 15

# sin(x) * cos(y) training data minimum error: 3.3511%, learning rate: 0.03, epochs: 2000, 2 hidden layer, hidden neurons: 40, 20
# sin(x) * cos(y) testing data minimum error: 3.9836%, learning rate: 0.03, epochs: 2000, 2 hidden layer, hidden neurons: 40, 20


# outputs = []
# errors = []
# x1 = []
# x2 = []
# sin = []
# sumerror = 0
# for x in range(len(training_data)):
	# outputs.append([])
	# errors.append([])
	# for y in range(len(training_data[x])):
		# outputs[x].append(nn.run_input(training_data[x][y][0], is_training=False))
		# errors[x].append(abs(outputs[x][y][0][0] - training_data[x][y][1][0]))
		# if x == y:
			# x1.append(training_data[x][y][0][0])
			# x2.append(training_data[x][y][0][1])
			
		# sumerror += abs(errors[x][y])
		# # print(f"Input: {training_data[x][y][0][0][0].ravel()[0]}, {training_data[x][y][0][1][0]}, Output: {outputs[x][y].ravel()[0]:.16f}, Target: {training_data[x][y][1][0][0].ravel()[0]}, Error: {errors[x][y] / 2 * 100}")
		
	
# print(sumerror / (len(errors) * len(errors)) / 2 * 100)
	

# xsin = np.linspace(-2 * np.pi, 2 * np.pi, 500)
# target = np.sin(xsin)
# ax = plt.figure().add_subplot(projection='3d')
# plt.text(-9, 1.2, 'Average Error: {:.4f}'.format(sum(errors) / len(errors) / 2 * 100))
# ax.scatter(np.asarray(x1), np.asarray(x2), np.asarray(outputs), marker='o')
# ax.scatter(xsin, target, linewidth=1, marker='.')
# plt.show()

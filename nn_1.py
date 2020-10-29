from operator import ne
import sys
import os
import math
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90


class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Parameters
		----------
						num_layers : Number of HIDDEN layers.
						num_units : Number of units in each Hidden layer.
		'''
		self.weights = []
		self.biases = []
		self.outs = []
		self.relus = []

		for i in range(num_layers + 1):
			if i == 0:
				# first matrix
				mat = np.random.uniform(-1, 1, (NUM_FEATS, num_units))
				bias = np.random.uniform(-1, 1, num_units)
			elif i == num_layers:
				# last matrix
				mat = np.random.uniform(-1, 1, (num_units, 1))
				bias = np.random.uniform(-1, 1, 1)
			else:
				# hidden layer matrices
				mat = np.random.uniform(-1, 1, (num_units, num_units))
				bias = np.random.uniform(-1, 1, num_units)

			self.weights.append(mat)
			self.biases.append(bias)

		return

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Parameters
		----------
						X : Input to the network, numpy array of shape m x d
		Returns
		----------
						y : Output of the network, numpy array of shape m x 1
		'''
		self.outs = []
		self.relus = []
		# self.outs.append(X)

		count = len(self.weights)
		for i in range(count):
			if i == 0:
				net = (X @ self.weights[i]) + self.biases[i]
			else:
				net = (out @ self.weights[i]) + self.biases[i]

			if i != count - 1:
				out = np.maximum(net, 0)
				relu = np.where(net > 0, 1, 0)
			else:
				out = np.array(net)
				relu = np.ones(net.shape)

			self.outs.append(out)
			self.relus.append(relu)

		y_hat = self.outs[-1]
		# y_hat = y_hat.clip(min=1922, max=2011)
		# self.outs[-1] = y_hat
		return y_hat

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
						X : Input to the network, numpy array of shape m x d
						y : Output of the network, numpy array of shape m x 1
						lamda : Regularization parameter.

		Returns
		----------
						del_w : derivative of loss w.r.t. all weight values (a list of matrices).
						del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		del_w = []
		del_b = []
		m = X.shape[0]
		count = len(self.weights)

		# dL/dw{i} = (out{i-1}.T @ ([temp{i-1} @ weight{i+1}.T] * R{i})) + 2*lamda*weight{i}

		temp = (2/m)*(self.outs[-1] - y)
		# temp = temp * self.relus[-1]	#no activation for last layer
		dw = (self.outs[-2].T @ temp) + 2*lamda*self.weights[-1]
		del_w.append(dw)
		db = np.sum(temp, axis=0) + 2*lamda*self.biases[-1]
		del_b.append(db)

		for i in range(count-2, -1, -1):

			temp = (temp @ self.weights[i+1].T) * self.relus[i]

			if i != 0:
				dw = self.outs[i-1].T @ temp
			else:
				dw = X.T @ temp

			dw = dw + 2*lamda*self.weights[i]
			del_w.append(dw)
			db = np.sum(temp, axis=0) + 2*lamda*self.biases[i]
			del_b.append(db)

		del_w.reverse()
		del_b.reverse()

		return del_w, del_b


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Stochastic Gradient Descent (SGD) based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.lr = learning_rate
		return

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
						weights: Current weights of the network.
						biases: Current biases of the network.
						delta_weights: Gradients of weights with respect to loss.
						delta_biases: Gradients of biases with respect to loss.
		'''
		new_weights = []
		new_biases = []

		for i in range(len(weights)):
			new_weights.append(weights[i] - self.lr * delta_weights[i])
			new_biases.append(biases[i] - self.lr * delta_biases[i])

		return new_weights, new_biases


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
					y : targets, numpy array of shape m x 1
					y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
					MSE loss between y and y_hat.
	'''
	mse_loss = np.mean(np.power(y_hat - y, 2))
	return mse_loss


def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
					weights and biases of the network.

	Returns
	----------
					l2 regularization
	'''
	# might be wrong
	l2_reg = 0
	for i in range(len(weights)):
		l2_reg += np.sum(np.power(weights[i], 2))
		l2_reg += np.sum(np.power(biases[i], 2))

	return l2_reg


def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
					y : targets, numpy array of shape m x 1
					y_hat : predictions, numpy array of shape m x 1
					weights and biases of the network
					lamda: Regularization parameter

	Returns
	----------
					l2 regularization loss
	'''
	loss = loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)
	return loss


def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
					y : targets, numpy array of shape m x 1
					y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
					RMSE between y and y_hat.
	'''
	return math.sqrt(loss_mse(y, y_hat))


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
					1. Run gradient descent algorithm for `max_epochs` epochs.
					2. For each bach of the training data
									1.1 Compute gradients
									1.2 Update weights and biases using step() of optimizer.
					3. Compute RMSE on dev data after running `max_epochs` epochs.
	'''
	epoch = 0
	total = train_input.shape[0]
	max_ins = (total//batch_size) if (total %
									  batch_size == 0) else (total//batch_size + 1)
	while(epoch < max_epochs):
		start = 0
		end = batch_size if batch_size < total else total
		ins = 0
		print("EPOCH",epoch)
		while(ins < max_ins):
			# print("TRAIN>>", epoch, ":", ins)
			# print("batch:", start, "to", end)
			# print("===========")
			X = train_input[start:end]
			y_hat = net(X)
			y = train_target[start:end]
			# print(y_hat.shape)
			# print(y.shape)
			# print("loss:", loss_mse(y, y_hat))
			del_w, del_b = net.backward(X, y, lamda)
			net.weights, net.biases = optimizer.step(
				net.weights, net.biases, del_w, del_b)

			# next instance
			start = end
			end = (end+batch_size) if (end+batch_size) < total else total
			ins += 1

		print("++++++++++++++++++")
		y_hat = net(train_input)
		y = train_target
		print("TRAIN LOSS:",rmse(y, y_hat))
		y_hat = net(dev_input)
		y = dev_target
		print("DEV LOSS:",rmse(y, y_hat))

		epoch += 1
		# sys.exit()
	
	print("train predictions")
	print(np.hstack((train_target, net(train_input))))
	print("dev predictions")
	print(np.hstack((dev_target, net(dev_input))))
	return


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
					net : trained neural network
					inputs : test input, numpy array of shape m x d

	Returns
	----------
					predictions (optional): Predictions obtained from forward pass
																													on test data, numpy array of shape m x 1
	'''
	predictions = net(inputs)
	return predictions


def read_data():
	'''
	Read the train, dev, and test datasets
	'''

	train_input = pd.read_csv(
		'dataset/train.csv', skipinitialspace=True).to_numpy()
	train_target = train_input[:, 0].astype(int)
	train_target = train_target.reshape(train_target.shape[0], 1)
	train_input = train_input[:, 1:].astype(float)
	# train_target = pd.read_csv('dataset/train.csv', skipinitialspace=True, usecols=['label']).to_numpy()

	dev_input = pd.read_csv(
		'dataset/dev.csv', skipinitialspace=True).to_numpy()
	dev_target = dev_input[:, 0].astype(int)
	dev_target = dev_target.reshape(dev_target.shape[0], 1)
	dev_input = dev_input[:, 1:].astype(float)
	# dev_target = pd.read_csv('dataset/dev.csv', skipinitialspace=True, usecols=['label']).to_numpy()

	test_input = pd.read_csv(
		'dataset/test.csv', skipinitialspace=True).to_numpy()
	test_input = test_input.astype(float)

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128

	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0  # Regularization Parameter

	print("reading dataset...")
	train_input, train_target, dev_input, dev_target, test_input = read_data()
	NUM_FEATS = train_input.shape[1]
	# print(train_target.shape)
	# print(train_target[0:3])
	# print(train_input[0:3].shape)

	print("initializing neural network...")
	net = Net(num_layers, num_units)

	# print("done")
	# print(len(net.weights))
	# for i in range(len(net.weights)):
	# 	print(net.weights[i].shape)
	# print(net.weights)

	# print(len(net.biases))
	# for i in range(len(net.biases)):
	# 	print(net.biases[i].shape)
	# print(net.biases)
	# print("lol")
	# sys.exit()

	optimizer = Optimizer(learning_rate)
	print("training neural network...")
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()

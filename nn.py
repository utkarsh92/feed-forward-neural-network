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
		self.biases = []
		self.weights = []

		for i in range(num_layers + 1):
			if i == 0:
				#weight after input layer
				mat = np.random.uniform(-1,1,(NUM_FEATS,num_units))
				bias = np.random.uniform(-1,1,num_units)
			elif i == num_layers:
				#weight before output layer
				mat = np.random.uniform(-1,1,(num_units,1))
				bias = np.random.uniform(-1,1,1)
			else:
				#weights in b/w hidden layer
				mat = np.random.uniform(-1,1,(num_units,num_units))
				bias = np.random.uniform(-1,1,num_units)
			
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
		for i in range(len(self.weights)):
			
			if i == 0:
				y_hat = np.matmul(X, self.weights[i]) + self.biases[i]
			else:
				y_hat = np.matmul(y_hat, self.weights[i]) + self.biases[i]

			y_hat = np.where(y_hat<0,0,y_hat)

		y_hat = y_hat.astype(int)
		y_hat = y_hat.clip(min=1922,max=2011)
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
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		del_W = []
		del_b = []

		return


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
		return


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
	mse_loss = np.mean(np.power(y_hat-y,2))
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
	return

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
	return math.sqrt(loss_mse(y,y_hat))


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
	return

def read_data():
	'''
	Read the train, dev, and test datasets
	'''

	train_input = pd.read_csv('dataset/train.csv', skipinitialspace=True).to_numpy()
	train_target = train_input[:,0].astype(int)
	train_input = train_input[:,1:].astype(float)
	# train_target = pd.read_csv('dataset/train.csv', skipinitialspace=True, usecols=['label']).to_numpy()

	dev_input = pd.read_csv('dataset/dev.csv', skipinitialspace=True).to_numpy()
	dev_target = dev_input[:,0].astype(int)
	dev_input = dev_input[:,1:].astype(float)
	# dev_target = pd.read_csv('dataset/dev.csv', skipinitialspace=True, usecols=['label']).to_numpy()

	test_input = pd.read_csv('dataset/test.csv', skipinitialspace=True).to_numpy()
	# test_input = test_input[:,1:].astype(float)

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128

	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	NUM_FEATS = train_input.shape[1]
	# print(test_input.shape)
	# print(train_input.shape[1])

	net = Net(num_layers, num_units)

	print(len(net.weights))
	for i in range(len(net.weights)):
		print(net.weights[i].shape)
	# print(net.weights)

	print(len(net.biases))
	for i in range(len(net.biases)):
		print(net.biases[i].shape)
	# print(net.biases)
	sys.exit()


	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()

"""Perceptron class implemetation
This is a classifying algorithm that attempts learn a binary classification task

The following is an attempt to visualize this algorithm 

Output is the binary classification of the inputs
Error happens when the output does not match the correct answer
Update Weight updates the weight by this function : Learning rate*(correct_answer - actual_answer)*X     # With X being the features
Activation Function takes the value of the Net Input Function and evaluates the output as follows: 
(output >=0 ? 1 else 0)                                                                                  # This is def predict
Net Input Function takes input values x and a weight vector w and z is the net input                     # this is net_input
w are the weights
x are the features

1----w0----|          (Update Weight) <-----ERROR<-----------------|
x1---w1---------(Net Input Function)----(Activation Function)----->Output
x2---w2----|
x3---w3----|
"""
import numpy as np
class Perceptron(object):
	"""Perceptron classifier.

	Parameters
	----------
	eta: float
		Learning rate (between 0.0 and 1.0)
	n_iter: int
		Passes over the training dataset.
	
	Attributes
	----------
	w_: 1d-array
		Weights after fitting
	errors_: list
		Number of misclassificatiosn in every epoch

	"""
	def __init__(self, eta=0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		"""Fit training data.

		Parameters
		----------
		X: {array-like}, shape = [n_samples, n_features]
			Training vectors, where n_samples
			is the number of samples and n_features is the number of features.
		y: array-like, shape = [n_samples]
			Target values.

		Returns
		-------
		self: object

		"""
		self.w_ = np.zeros( 1 + X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self
	
	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)

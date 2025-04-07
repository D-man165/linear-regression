# Building a Lnear Regression model from scratch without use of Scikit-Learn using Gradient Descent and MSE Cost function
import numpy as np
import math


class LinearRegressionCustom:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lrate = learning_rate
        self.niters = iterations
        self.weights = 0
        self.bias = 0
        self.samples = 1
        self.features = 1

    def cost_function(self, X, y):
        """
        Measures deviation from known dataset (y)
        """
        # better than matrix multiplicaion "@" for vectors
        y_pred = np.dot(X, self.weights) + self.bias
        cost = np.mean(np.square(y_pred - y))
        return cost
    
    def gradient_descent(self, X, y):
        """
        Returns the changes needed in values of m and c
        Below formulae are easily calculated through partial derivatives of cost function
        """
        y_pred = np.dot(X, self.weights) + self.bias
        dw = 2 * (1/self.samples) * np.dot(X.T, (y_pred - y))
        db = 2 * np.mean(y_pred - y)
        return dw, db
    
    def fit(self, X, y):
        """
        Fits the training data and uses above two functions to adjust values of m and c
        """
        if len(X.shape) == 1:
            self.samples = X.shape[0]
            self.features = 1
        else:
            self.samples, self.features = X.shape
        self.weights = np.zeros((self.features, 1))
        self.bias = 0
        prev_cost = math.inf  # Starts with a large value for previous cost

        for iter in range(self.niters):
            dw, db = self.gradient_descent(X, y)
            self.weights -= self.lrate * dw
            self.bias -= self.lrate * db
            cost = self.cost_function(X, y)
            if iter % 100 == 0:  # every 100 epochs
                print(f"Iteration {iter}: m = {np.round(self.weights, 4)}, c = {round(self.bias, 4)}, cost = {round(cost, 4)}")
            if abs(prev_cost - cost) < 0.0001:  # if cost has conferges
                print("Model has converged")
                print(f"Iteration {iter}: m = {np.round(self.weights, 4)}, c = {round(self.bias, 4)}, cost = {round(cost, 4)}")
                return    
            prev_cost = cost

    def predict(self, X):
        """
        Returns the predicted array
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def evaluate(self, y, y_pred):
        mape = np.mean(abs(y - y_pred)/y)
        return mape
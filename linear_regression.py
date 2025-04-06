# Building a Lnear Regression model from scratch without use of Scikit-Learn using Gradient Descent and MSE Cost function
import numpy as np
import math

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lrate = learning_rate
        self.niters = n_iters
        self.weights = 0
        self.bias = 0
        self.samples = 1
        self.features = 1

    def cost_function(self, y_train, x_train):
        """
        Measures deviation from known dataset(y_train)
        """
        y_pred = x_train @ self.weights.T + self.bias
        cost = np.mean(np.square(y_pred - y_train))
        return cost
    
    def gradient_descent(self, y_train, x_train):
        """
        Returns the changes needed in values of m and c
        Below formulae are easily calclated through partial derivatives of cost function
        """
        y_pred = x_train @ self.weights.T + self.bias
        dw = 2 * (1/self.samples) * (x_train.T @ (y_pred - y_train))
        db = 2 * np.mean(y_pred - y_train)
        return dw, db
    
    def fit(self, y_train, x_train):
        """
        Fits the training data and uses above two functions to adjust values of ma and c
        """
        self.samples, self.features = x_train.shape
        self.weights = np.zeros(self.features)
        self.bias = 0

        prev_cost = math.inf # begins with a large value for previous cost value
        for iter in range(self.niters):
            dw, db = self.gradient_descent(y_train, x_train)
            self.weights -= self.lrate * dw
            self.bias -= self.lrate * db
            cost = self.cost_function(y_train, x_train)
            if abs(prev_cost - cost) < 0.0001: # in case of stagnancy
                print("Model has converged")
                print(f"Iteration {iter}: m = {np.round(self.weights, 4)}, c = {round(self.bias, 4)}, cost = {round(cost, 4)}")
                return
            if iter % 100 == 0: # once every 100 epochs
                print(f"Iteration {iter}: m = {np.round(self.weights, 4)}, c = {round(self.bias, 4)}, cost = {round(cost, 4)}")
            prev_cost = cost

    def predict(self, x_test):
        """
        Returns the predicted array
        """
        y_pred = x_test @ self.weights.T + self.bias
        return y_pred
    
    def evaluate(self, y_test, x_test):
        """
        Returns the Mean Absolute Percentage Error
        very good for relative error
        """
        y_pred = self.predict(x_test)
        mape = np.mean(abs(y_test - y_pred)/y_test)
        return mape
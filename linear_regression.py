# Building a Lnear Regression model from scratch without use of Scikit-Learn using Gradient Descent and MSE Cost function
import numpy as np
import math

class LinearRegression:
    def __init__(self, learning_rate=0.02, n_iters=1000):
        self._lrate = learning_rate
        self._niters = n_iters
        self._m = 0
        self._c = 0

    def cost_function(self, y_train, x_train):
        # measures deviation from known dataset(y_train)
        y_pred = self._m * x_train + self._c
        cost = np.mean(np.square(y_pred - y_train))
        return cost
    
    def gradient_descent(self, y_train, x_train):
        # returns the changes needed in values of m and c
        y_pred = self._m * x_train + self._c
        # below formulae are easily calclated through partial derivatives of cost function
        dw = 2 * np.mean(x_train * (y_pred - y_train)) # with respect to the weight(m)
        db = 2 * np.mean(y_pred - y_train) # with respect to the bias(c)
        return dw, db
    
    def fit(self, y_train, x_train):
        # fits the training data and uses above two functions to adjust values of ma and c
        prev_cost = math.inf # begins with a large value for previous cost value
        for iter in range(self._niters):
            dw, db = self.gradient_descent(y_train, x_train)
            self._m -= self._lrate * dw
            self._c -= self._lrate * db
            cost = self.cost_function(y_train, x_train)
            if abs(prev_cost - cost) < 0.00001: # in case of stagnancy
                print("Model has converged")
                print(f"Iteration {iter}: m = {round(self._m, 4)}, c = {round(self._c, 4)}, cost = {round(cost, 4)}")
                return
            if iter % 100 == 0: # once every 100 epochs
                print(f"Iteration {iter}: m = {round(self._m, 4)}, c = {round(self._c, 4)}, cost = {round(cost, 4)}")
            prev_cost = cost

    def predict(self, x_test):
        # returns the predicted array
        y_pred = self._m * x_test + self._c
        return y_pred
    
    def evaluate(self, y_test, x_test):
        # returns the Mean Absolute Percentage Error
        # very good for relative error
        y_pred = self.predict(x_test)
        mape = np.mean(abs(y_test - y_pred)/y_test)
        return mape
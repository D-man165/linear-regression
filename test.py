from linear_regression import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

X = np.random.rand(100, 1)
noise = np.random.uniform(0, 1,(100,1))
y = 2 + 3 * X + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# custom model
model = LinearRegressionCustom()
model.fit(X_train, y_train)
y_pred_custom = model.predict(X_test)

# scikit-learn model
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

mape_custom = round(model.evaluate(y_test, y_pred_custom), 4)
mape_sklearn = round(mean_absolute_percentage_error(y_test, y_pred_sklearn), 4)

mse_custom = round(mean_squared_error(y_test, y_pred_custom), 4)
mse_sklearn = round(mean_squared_error(y_test, y_pred_sklearn), 4)

print("Mean Absolute Percentage Error custom:", mape_custom)
print("Mean Absolute Percentage Error sklearn:", mape_sklearn)
print("Mean Squared Error custom:", mse_custom)
print("Mean Squared Error sklearn:", mse_sklearn)
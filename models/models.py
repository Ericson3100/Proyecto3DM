import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
# Modelo1: Regresión lineal con ecuación normal, se pasan un dataframe con las columnas
class LinearRegressionNormalEquation:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Ensure y is a column vector
        y = np.array(y).reshape(-1, 1)
        # Normal equation: theta = (X^T * X)^-1 * X^T * y
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

#Modelo2: Regresión lineal con sklearn

class LinearRegressionSklearn:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
#Modelo3: Regresion lineal con SVD (Singular Value Decomposition) de numpy

class LinearRegressionSVD:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Ensure y is a column vector
        y = np.array(y).reshape(-1, 1)
        # Compute the pseudoinverse of X_b using np.linalg.pinv
        X_b_pinv = np.linalg.pinv(X_b)
        # Compute theta
        self.theta = X_b_pinv.dot(y)

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
    
#Modelo4: Regresión polinomial usando pipeline de sklearn, PolynomialFeatures y LinearRegression,
# se pasa el grado del polinomio como parámetro

class PolynomialRegression:
    def __init__(self, degree):
        #No se incluye el bias al generar las características polinómicas
        self.model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin_reg", LinearRegression())
        ])
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
#Modelo5: Regresión lineal con descenso de gradiente Batch Gradient Descent
#usando numpy, se pasan el learning rate y el número de iteraciones
class LinearRegressionBatchGradientDescent:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
        # Initialize theta with random values
        np.random.seed(42)
        self.theta = np.random.randn(X_b.shape[1], 1)
        # Perform n_iterations of gradient descent
        for iteration in range(self.n_iterations):
            gradients = 1 / X_b.shape[0] * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
#Modelo6: Regresión lineal con descenso de gradiente Batch Gradient Descent
#ajustando los parametros de SGD de sklearn

class LinearRegressionBGDSklearn:
    def __init__(self, learning_rate, n_iterations):
        self.model = SGDRegressor(eta0=learning_rate, max_iter=n_iterations, tol=None, penalty=None,
                                  learning_rate="constant", shuffle=False)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

#Modelo7: Regresión lineal con descenso de gradiente Stochastic Gradient Descent
#usando numpy, se pasan el learning rate y el número de iteraciones
class LinearRegressionStochasticGradientDescent:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y=np.array(y).reshape(-1, 1)  # Ensure y is a column vector
        # Initialize theta with random values
        np.random.seed(42)
        self.theta = np.random.randn(X_b.shape[1], 1)
        # Perform n_iterations of gradient descent
        for iteration in range(self.n_iterations):
            random_index = np.random.randint(X_b.shape[0])
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = xi.T.dot(xi.dot(self.theta) - yi)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)
#Modelo8: Regresión lineal con descenso de gradiente Stochastic Gradient Descent
#ajustando los parametros de SGD de sklearn
class LinearRegressionSGDSklearn:
    def __init__(self, learning_rate, n_iterations):
        self.model = SGDRegressor(eta0=learning_rate, max_iter=n_iterations, tol=None, penalty=None,
                                  learning_rate="constant", shuffle=True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
#Modelo9: Regresión lineal con regularización Lasso usando sklearn

class LassoRegression:
    def __init__(self, alpha):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
#Modelo10: Regresión lineal con regularización Ridge usando sklearn
#Reguralización L2 es más suave que L1. Regresión L1 anula totalmente
#los coeficientes de las características menos importantes mientras que L2 mantiene con un valor muy pequeño

class RidgeRegression:
    def __init__(self, alpha):
        self.model = Ridge(alpha=alpha,solver='svd')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    